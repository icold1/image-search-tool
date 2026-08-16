"""Chinese-CLIP 模型封装：懒加载、GPU/CPU 自适应、批量编码。

使用 cn_clip 官方包。首次调用 load_model() 会下载约 1.63GB 权重
（优先 ModelScope，失败回退 HuggingFace）到 data/models/checkpoints/。
"""
# 必须先于 import torch：注册 conda 环境的 DLL 目录（cuDNN 等）
from core._win import ensure_env_dlls
ensure_env_dlls()

import threading
from typing import List, Sequence

import numpy as np
import torch

_lock = threading.Lock()
_model = None
_preprocess = None
_device = "cpu"


def _install_flash_attn_stub() -> None:
    """cn_clip 1.6.0 会硬导入 flash_attn.flash_attention（仅训练用）。

    本机 flash-attn 2.8.3 已无该模块（重构移除），且 Windows 上也不可用；
    在 sys.modules 注入桩模块，让 importlib.import_module 直接命中缓存。
    官方推理权重 use_flash_attention=False，桩永远不会被实例化。
    """
    import sys as _sys
    import types as _types
    if "flash_attn.flash_attention" not in _sys.modules:
        stub = _types.ModuleType("flash_attn.flash_attention")

        class _UnavailableFlashMHA:
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "flash_attn 在本机不可用；推理路径不受影响")

        stub.FlashMHA = _UnavailableFlashMHA
        _sys.modules["flash_attn.flash_attention"] = stub


def get_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def load_model(model_name: str = "ViT-L-14", device: str = "auto",
               model_dir: str = ""):
    """加载模型（线程安全、幂等）。返回模型对象（供 model_manager 持有）。"""
    global _model, _preprocess, _device
    with _lock:
        if _model is not None:
            return _model
        _install_flash_attn_stub()
        import cn_clip.clip as clip  # 延迟导入，避免拖慢 UI 启动
        dev = get_device(device)
        # 权重下载：优先 ModelScope（国内快），失败回退 HuggingFace。
        # 若本地已存在权重文件（download_root/模型名.pt），二者都会被跳过。
        try:
            model, preprocess = clip.load_from_name(
                model_name, device=dev,
                download_root=str(model_dir) if model_dir else None,
                use_modelscope=True)
        except Exception as e:
            if "modelscope" in str(e).lower():
                model, preprocess = clip.load_from_name(
                    model_name, device=dev,
                    download_root=str(model_dir) if model_dir else None,
                    use_modelscope=False)
            else:
                raise
        model.eval()
        _model, _preprocess, _device = model, preprocess, dev
        return model


def unload() -> None:
    """卸载模型并回收显存（线程安全）。

    编码函数入口都先捕获本地引用，卸载不会导致进行中的编码崩溃。
    """
    global _model, _preprocess, _device
    with _lock:
        _model = None
        _preprocess = None
        _device = "cpu"
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_loaded() -> bool:
    return _model is not None


def encode_text(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """文本 -> L2 归一化特征 (n, dim)。"""
    model = _model  # 捕获本地引用：期间被 unload 也不受影响
    device = _device
    if model is None:
        raise RuntimeError("模型未加载，请先调用 load_model()")
    import cn_clip.clip as clip
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = clip.tokenize(batch).to(device)
            f = model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def encode_images(images, batch_size: int = 64) -> np.ndarray:
    """PIL 图像（或列表）-> L2 归一化特征 (n, dim)。"""
    model = _model  # 捕获本地引用：期间被 unload 也不受影响
    preprocess = _preprocess
    device = _device
    if model is None:
        raise RuntimeError("模型未加载，请先调用 load_model()")
    if not isinstance(images, (list, tuple)):
        images = [images]
    images = list(images)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([preprocess(img) for img in batch]).to(device)
            f = model.encode_image(tensors)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def feature_dim() -> int:
    """当前模型的特征维度。"""
    if _model is None:
        raise RuntimeError("模型未加载，请先调用 load_model()")
    try:
        return int(_model.visual.output_dim)
    except AttributeError:
        from PIL import Image
        return int(encode_images([Image.new("RGB", (64, 64))]).shape[1])
