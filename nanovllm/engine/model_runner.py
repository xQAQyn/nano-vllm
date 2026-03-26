import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.speculative_sampler import SpeculativeSampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2753", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # EAGLE speculative decoding support
        self.eagle_enabled = config.eagle_enabled
        self.speculation_depth = config.speculation_depth
        self.draft_model = None
        self.speculative_sampler = None
        if self.eagle_enabled:
            from nanovllm.models.eagle import load_draft_model
            from nanovllm.engine.eagle_runner import EagleDraftRunner
            self.draft_model = load_draft_model(
                hf_config, self.model, draft_model_path=config.eagle_draft_model
            )
            self.draft_model.eval()
            self.draft_runner = EagleDraftRunner(
                draft_model=self.draft_model,
                max_speculation_depth=self.speculation_depth,
            )
            self.speculative_sampler = SpeculativeSampler()
        
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager and hasattr(self, 'graphs'):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        # Only destroy process group if it was initialized
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # Use EAGLE speculative decoding if enabled and not in prefill phase
        if self.eagle_enabled and not is_prefill and len(seqs) == 1:
            return self.run_eagle(seqs)
        else:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill)
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            reset_context()
            return token_ids

    def run_eagle(self, seqs: list[Sequence]) -> list[int]:
        """Run EAGLE speculative decoding for a single sequence.

        Steps:
        1. Run target model to get hidden states from second-to-last layer
        2. Use draft model to generate K draft tokens autoregressively
        3. Run target model once with all draft tokens to get verification logits
        4. Use speculative sampling to accept/reject draft tokens
        5. Return accepted tokens (or resampled token on rejection)
        """
        assert len(seqs) == 1, "EAGLE currently supports single sequence only"
        seq = seqs[0]

        # Prepare decode input
        input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        temperature = temperatures[0].item() if temperatures is not None else 1.0

        # Run target model to get hidden states and logits for current position
        # We need hidden states from second-to-last layer for draft model
        with torch.inference_mode():
            # Get logits for current token (to compute draft acceptance later)
            current_logits = self.run_model(input_ids, positions, is_prefill=False)

            # Get hidden states from target model for draft generation
            # Run model with return_hidden_states=True
            hidden_states, second_to_last_hidden = self.model(
                input_ids, positions, return_hidden_states=True
            )
            reset_context()

        # Generate draft tokens using draft model
        # Use the second-to-last hidden states as input to draft model
        with torch.inference_mode():
            draft_token_ids, draft_features = self.draft_runner.generate_draft_tokens(
                hidden_states=second_to_last_hidden,
                token_ids=input_ids,
                positions=positions,
                num_draft_tokens=self.speculation_depth,
            )

        # Now run target model with draft tokens to get verification logits
        # Append draft tokens to input for verification
        draft_input_ids = torch.cat([
            input_ids,
            torch.tensor(draft_token_ids, dtype=input_ids.dtype, device=input_ids.device)
        ])
        draft_positions = torch.arange(
            positions[0].item(),
            positions[0].item() + len(draft_input_ids),
            dtype=positions.dtype,
            device=positions.device,
        )

        # Prepare context for extended sequence (verification phase)
        # This is a prefill operation - we need to compute KV for all tokens
        seq_len = len(draft_input_ids)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=input_ids.device)
        
        # Compute slot_mapping for the extended sequence
        # For verification, we need to store KV cache for all positions
        # The sequence already has context_len tokens, and we're adding draft tokens
        # We need to store KV at the correct positions in the cache
        slot_mapping = []
        context_len = len(seq)  # Current sequence length before adding draft tokens
        for i in range(seq_len):
            # Position in the full sequence
            pos = context_len - 1 + i  # Start from last position
            # Compute block index and offset within block
            block_idx = pos // self.block_size
            offset_in_block = pos % self.block_size
            # Get the physical block ID from block table
            if block_idx < len(seq.block_table):
                physical_block = seq.block_table[block_idx]
                slot = physical_block * self.block_size + offset_in_block
            else:
                # Fallback: use last block (should not happen in normal operation)
                slot = seq.block_table[-1] * self.block_size + offset_in_block
            slot_mapping.append(slot)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=input_ids.device)
        
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=slot_mapping,
        )

        with torch.inference_mode():
            # Run target model on full sequence including draft tokens
            full_hidden = self.model(draft_input_ids, draft_positions)
            full_logits = self.model.compute_logits(full_hidden)
            reset_context()

        # Extract logits for draft token positions (positions after current)
        # draft_logits_start is the position where draft tokens begin
        draft_logits_start = len(input_ids)
        
        # Safety check: ensure we have enough logits
        if full_logits.shape[0] <= draft_logits_start:
            # Fallback: no draft logits available, sample from current position
            result_tokens = self.sampler(
                current_logits, temperatures
            ).tolist() if self.rank == 0 else []
            return result_tokens
        
        target_logits_for_draft = full_logits[draft_logits_start:]  # [K, vocab_size]
        
        # Safety check: ensure target_logits_for_draft has the right shape
        if target_logits_for_draft.shape[0] != len(draft_token_ids):
            # Mismatch - use what we have or fallback
            if target_logits_for_draft.shape[0] == 0:
                result_tokens = self.sampler(
                    current_logits, temperatures
                ).tolist() if self.rank == 0 else []
                return result_tokens
            # Truncate draft tokens to match available logits
            draft_token_ids = draft_token_ids[:target_logits_for_draft.shape[0]]

        # Get draft model logits for the same positions
        # We need to run draft model again to get logits for all draft positions
        with torch.inference_mode():
            # Prepare draft inputs for all positions
            if len(draft_token_ids) > 1:
                draft_token_ids_tensor = torch.tensor(
                    draft_token_ids[:-1], dtype=input_ids.dtype, device=input_ids.device
                )  # All but last draft token
                draft_hidden_input = second_to_last_hidden[-1:].expand(len(draft_token_ids) - 1, -1)
                draft_pos_input = torch.arange(
                    draft_logits_start,
                    draft_logits_start + len(draft_token_ids) - 1,
                    dtype=positions.dtype,
                    device=positions.device,
                )

                cu_seqlens_draft = torch.tensor(
                    [0, len(draft_token_ids) - 1], dtype=torch.int32, device=input_ids.device
                )
                set_context(
                    is_prefill=True,
                    cu_seqlens_q=cu_seqlens_draft,
                    cu_seqlens_k=cu_seqlens_draft,
                    max_seqlen_q=len(draft_token_ids) - 1,
                    max_seqlen_k=len(draft_token_ids) - 1,
                )
                _, draft_logits = self.draft_model(
                    token_ids=draft_token_ids_tensor,
                    hidden_states=draft_hidden_input,
                    positions=draft_pos_input,
                )
                reset_context()
            else:
                draft_logits = target_logits_for_draft[:1]  # Fallback for single draft token

        # Perform speculative sampling to verify draft tokens
        # Use non-compiled version to avoid torch.compile issues
        with torch.inference_mode():
            accepted_tokens, accepted_mask, resampled_token = self.speculative_sampler.verify_tokens(
                target_logits=target_logits_for_draft,
                draft_logits=draft_logits,
                draft_token_ids=draft_token_ids,
                temperature=temperature,
            )
        
        # Update sequence state with accepted tokens
        for i, (token_id, accepted) in enumerate(zip(draft_token_ids, accepted_mask)):
            if accepted:
                seq.append_token(token_id)
                seq.accepted_mask.append(True)
            else:
                seq.accepted_mask.append(False)
                # Add resampled token on rejection
                if resampled_token is not None:
                    seq.append_token(resampled_token)
                break
        
        # Clear draft state
        seq.clear_draft()
        
        # Return the last generated token(s)
        # For compatibility with scheduler, return the last token
        result_tokens = []
        if accepted_mask and any(accepted_mask):
            # Return all accepted tokens
            result_tokens = accepted_tokens
            if resampled_token is not None and not all(accepted_mask):
                result_tokens.append(resampled_token)
        else:
            # All rejected, use resampled token
            if resampled_token is not None:
                result_tokens = [resampled_token]
            else:
                # Fallback: sample from current logits
                result_tokens = self.sampler(
                    current_logits, temperatures
                ).tolist() if self.rank == 0 else []
        
        return result_tokens

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
