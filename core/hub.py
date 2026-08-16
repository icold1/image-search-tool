"""模型权重获取：本地路径优先，ModelScope 下载优先，失败回退 HuggingFace。

与 core/model.py 的 cn_clip 下载逻辑互补；供 transformers 系列的
VLM / 文本嵌入模型使用。
"""
from pathlib import Path


def resolve_pretrained(name: str, model_dir: str = "") -> str:
    """返回可直接传给 from_pretrained 的路径或 repo id。

    优先级：本地目录 > ModelScope snapshot_download > 原始 name（HF）。
    任何下载失败都静默回退，让 transformers 走 HF 默认下载。
    """
    p = Path(name)
    if p.is_dir():
        return str(p)
    if model_dir:
        # 约定：model_dir/组织--模型名（与 cn_clip 的 download_root 风格一致）
        local = Path(model_dir) / name.replace("/", "--")
        if local.is_dir():
            return str(local)
    try:
        from modelscope import snapshot_download
        path = snapshot_download(name)
        if path and Path(path).is_dir():
            return str(path)
    except Exception:
        pass
    return name
