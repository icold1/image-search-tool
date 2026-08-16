"""Qwen3-Embedding 文本嵌入封装：caption 行向量 + 查询向量。

- 查询侧显式加官方指令前缀（Qwen3-Embedding 为指令感知模型，
  参考 https://huggingface.co/Qwen/Qwen3-Embedding-0.6B）。
- MRL 截断：默认取前 embed_dim 维（1024 全量 / 可降 512 省内存），
  截断后重新 L2 归一化。
- 经 model_manager 管理（slot "embed"），与 CLIP 共用空闲卸载策略。
"""
import threading
from typing import List, Sequence

import numpy as np

QUERY_INSTRUCTION = (
    "Instruct: 给定一个用户查询，检索与之最相关的图片描述\nQuery: {query}")

_lock = threading.Lock()
_model = None
_tokenizer = None
_dev = "cpu"


def _resolve_device(pref: str) -> str:
    import torch
    if pref == "cpu":
        return "cpu"
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def load(model_name: str = "Qwen/Qwen3-Embedding-0.6B",
         device: str = "auto", model_dir: str = "") -> None:
    """加载嵌入模型（线程安全、幂等）。"""
    global _model, _tokenizer, _dev
    with _lock:
        if _model is not None:
            return
        # 必须先于 import torch：注册 conda 环境 DLL 目录（cuDNN 等）
        from core._win import ensure_env_dlls
        ensure_env_dlls()
        import torch
        from transformers import AutoModel, AutoTokenizer
        from core.hub import resolve_pretrained
        dev = _resolve_device(device)
        name = resolve_pretrained(model_name, model_dir)
        dtype = torch.float16 if dev == "cuda" else torch.float32
        try:
            model = AutoModel.from_pretrained(
                name, dtype=dtype, trust_remote_code=True).eval().to(dev)
        except TypeError:
            # transformers < 4.56 用 torch_dtype
            model = AutoModel.from_pretrained(
                name, torch_dtype=dtype, trust_remote_code=True).eval().to(dev)
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        _model, _tokenizer, _dev = model, tokenizer, dev


def unload() -> None:
    global _model, _tokenizer, _dev
    with _lock:
        _model = None
        _tokenizer = None
        _dev = "cpu"


def is_loaded() -> bool:
    return _model is not None


def encode(texts: Sequence[str], is_query: bool = False, batch_size: int = 32,
           dim: int = 0) -> np.ndarray:
    """文本 -> L2 归一化特征 (n, dim|full)。dim>0 时做 MRL 截断再归一化。"""
    import torch
    import torch.nn.functional as F
    model = _model          # 捕获本地引用：期间被 unload 也不受影响
    tokenizer = _tokenizer
    dev = _dev
    if model is None:
        raise RuntimeError("嵌入模型未加载，请先调用 load()")
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    if is_query:
        texts = [QUERY_INSTRUCTION.format(query=t) for t in texts]
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=1024, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = F.normalize(emb, dim=-1)
            if dim and 0 < dim < emb.shape[1]:
                emb = F.normalize(emb[:, :dim], dim=-1)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def query_vector(query: str, model_name: str, device: str, dim: int,
                 model_dir: str = "") -> np.ndarray:
    """经 model_manager 获取嵌入模型并编码查询（单条）。"""
    from core import model_manager
    mgr = model_manager.ModelManager.get()
    idle = _idle_seconds()
    mgr.slot("embed", idle_seconds=idle).configure(
        lambda: load(model_name, device, model_dir), unload_fn=unload)
    if not is_loaded():
        mgr.acquire("embed")
    else:
        mgr.touch("embed")
    return encode([query], is_query=True, batch_size=1, dim=dim)[0]


def _idle_seconds() -> int:
    try:
        from core import config as cfg
        return int(cfg.load_config().get("idle_unload_seconds", 0))
    except Exception:
        return 0
