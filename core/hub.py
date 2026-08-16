"""模型权重获取：本地路径优先，ModelScope 下载优先，失败回退 HuggingFace。

与 core/model.py 的 cn_clip 下载逻辑互补；供 transformers 系列的
VLM / 文本嵌入 / 精排模型使用。

缓存位置固定重定向到项目 data/models/cache/ 下（用户未显式设置环境变量时
自动设置），避免每次运行都因默认用户目录缓存为空而重新下载。
"""
import os
from pathlib import Path


def _ensure_project_caches() -> None:
    """把模型缓存统一到项目 data/models/cache（显式设置过的环境变量优先）。"""
    root = Path(__file__).resolve().parent.parent
    cache_root = root / "data" / "models" / "cache"
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))


def resolve_pretrained(name: str, model_dir: str = "") -> str:
    """返回可直接传给 from_pretrained 的路径或 repo id。

    优先级：本地目录 > 项目内镜像目录 > ModelScope 缓存/下载 > 原始 name（HF）。
    任何下载失败都静默回退，让 transformers 走 HF 默认下载。
    """
    _ensure_project_caches()
    p = Path(name)
    if p.is_dir():
        return str(p)
    if model_dir:
        # 约定：model_dir/组织--模型名（与 cn_clip 的 download_root 风格一致）
        local = Path(model_dir) / name.replace("/", "--")
        if local.is_dir():
            return str(local)
    # 项目内 HF 完整仓库镜像（含模块化子目录，如 reranker 的 1_LogitScore/）
    hf_local = (Path(__file__).resolve().parent.parent / "data" / "models"
                / "cache" / "hf-models" / name.replace("/", "--"))
    if hf_local.is_dir():
        return str(hf_local)
    # ModelScope 项目缓存直查：命中直接返回（跳过 snapshot_download 的
    # 联网校验，重复运行零网络开销）；未命中才走下载
    ms_cache = (Path(os.environ.get("MODELSCOPE_CACHE", ""))
                / "models" / name.replace("/", "--"))
    if ms_cache.is_dir():
        snaps = sorted(ms_cache.glob("snapshots/*"))
        if snaps:
            return str(snaps[0])
    try:
        from modelscope import snapshot_download
        path = snapshot_download(name)
        if path and Path(path).is_dir():
            return str(path)
    except Exception:
        pass
    return name
