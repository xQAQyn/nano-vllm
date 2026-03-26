import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        # Clean up atexit handler
        try:
            atexit.unregister(self.exit)
        except Exception:
            pass

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """Execute a single step of the generation loop.

        For standard decoding:
        1. Schedule sequences (prefill or decode)
        2. Run model forward pass
        3. Postprocess results

        For EAGLE speculative decoding:
        1. Schedule sequences
        2. Run draft model to generate K draft tokens
        3. Run target model verification
        4. Postprocess with acceptance/rejection

        Returns:
            tuple:
                - outputs: List of (seq_id, token_ids) for finished sequences
                - num_tokens: Number of tokens processed (positive for prefill, negative for decode)
        """
        seqs, is_prefill = self.scheduler.schedule()

        # For EAGLE, pass speculation info to model runner
        if self.scheduler.eagle_enabled and not is_prefill:
            # EAGLE mode: model runner handles draft + verify internally
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            # For EAGLE, token_ids may be a list of lists (multiple tokens per seq)
            # Normalize to list[int] format for scheduler
            if token_ids and isinstance(token_ids[0], list):
                # Multiple tokens per sequence - take last token for compatibility
                last_tokens = [tokens[-1] if tokens else 0 for tokens in token_ids]
                self.scheduler.postprocess(seqs, last_tokens)
                # Collect all accepted tokens for output
                outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            else:
                self.scheduler.postprocess(seqs, token_ids)
                outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        else:
            # Standard mode
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        prefill_time = decode_time = 0.
        prefill_tokens = decode_tokens = 0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_tokens += num_tokens
                    prefill_time += perf_counter() - t
                    prefill_throughput = prefill_tokens / prefill_time
                else:
                    decode_tokens -= num_tokens
                    decode_time += perf_counter() - t
                    decode_throughput = decode_tokens / decode_time
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
