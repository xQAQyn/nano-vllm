import os
from huggingface_hub import snapshot_download

size = "1.7"

def download_model():
    # ---------------- 配置区域 ----------------
    
    # 1. Hugging Face 上的模型 ID
    # 注意：Qwen 系列通常是 0.5B。这里使用的是最新的 Qwen2.5-0.5B。
    # 如果你需要对话指令版，请改为 "Qwen/Qwen2.5-0.5B-Instruct"
    repo_id = f"Qwen/Qwen3-{size}B"
    
    # 2. 本地保存路径 (你指定的路径)
    local_dir = f"./models/Qwen3-{size}B"
    
    # ----------------------------------------

    print(f"准备从 {repo_id} 下载模型...")
    print(f"目标本地目录: {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            
            # 重要参数：设置为 False，确保下载的是真实的物理文件，而不是缓存链接
            # 这样你可以把这个文件夹直接拷贝到其他机器使用
            local_dir_use_symlinks=False,
            
            # 支持断点续传，如果网络中断，重新运行脚本即可
            resume_download=True,
            
            # 可选：如果只想下载 pytorch 权重，可以排除 safetensors (反之亦然)
            # 但为了兼容性，通常建议保持默认，下载所有文件
            # ignore_patterns=["*.msgpack", "*.h5"] 
        )
        print("\n✅ 下载成功！")
        print(f"文件已保存在: {os.path.abspath(local_dir)}")
        
    except Exception as e:
        print(f"\n❌ 下载出错: {e}")

if __name__ == "__main__":
    download_model()