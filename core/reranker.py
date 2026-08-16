"""Qwen3-Reranker 交叉编码精排（手动 LogitScore 实现）。

原理（与官方 sentence-transformers 的 LogitScore 模块一致）：基座
Qwen3-0.6B 因果模型 + 输入末尾位置 "true"（9693）/ "false"（2152）
两个 token 的 logit 差值作为相关性分——打分头无权重，模型文件只有
基座权重，无需依赖 ST 5.3+ 的模块化加载。

- 经 model_manager 槽位 "reranker" 管理：默认常驻（查询速度优先），
  caption 批次开始时由 pipeline 显式卸载腾显存。
- 失败/未启用时 pipeline 静默降级（不精排），不影响主检索。
"""
import threading
from typing import List, Sequence

import numpy as np

_lock = threading.Lock()
_model = None
_tokenizer = None
_dev = "cpu"

TRUE_TOKEN = 9693
FALSE_TOKEN = 2152

# 官方默认指令（训练时指令主要为英文，README 建议使用英文指令）；
# 仓库 chat_template.jinja 用 role="system" 携带指令、role="query"/
# role="document" 携带查询与文档，判定提示词由模板硬编码。
INSTRUCTION = ("Given a web search query, retrieve relevant passages "
               "that answer the query")


def load(model_name: str = "Qwen/Qwen3-Reranker-0.6B",
         device: str = "auto", model_dir: str = "") -> None:
    """加载精排模型（线程安全、幂等）。"""
    global _model, _tokenizer, _dev
    with _lock:
        if _model is not None:
            return
        from core._win import ensure_env_dlls
        ensure_env_dlls()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from core.hub import resolve_pretrained
        dev = "cuda" if (device in ("cuda", "auto")
                         and torch.cuda.is_available()) else "cpu"
        name = resolve_pretrained(model_name, model_dir)
        dtype = torch.float16 if dev == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, dtype=dtype, trust_remote_code=True).eval().to(dev)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, trust_remote_code=True).eval().to(dev)
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _model, _tokenizer, _dev = model, tokenizer, dev


def unload() -> None:
    global _model, _tokenizer, _dev
    with _lock:
        _model = None
        _tokenizer = None
        _dev = "cpu"


def is_loaded() -> bool:
    return _model is not None


def rerank(query: str, docs: Sequence[str],
           batch_size: int = 16) -> np.ndarray:
    """返回 docs 对应的相关性分数（P(true)=sigmoid(logit_true-logit_false)）。

    输入按官方用法包一层 chat template（system 裁判提示 + user 内嵌
    Instruct/Query/Document），末尾位置取 true/false token 的 logit。
    """
    import torch
    model = _model
    tokenizer = _tokenizer
    dev = _dev
    if model is None:
        raise RuntimeError("精排模型未加载，请先调用 load()")
    messages = [
        [{"role": "system", "content": INSTRUCTION},
         {"role": "query", "content": query},
         {"role": "document", "content": d}]
        for d in docs
    ]
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            ids = []
            for m in batch:
                try:
                    ids.append(tokenizer.apply_chat_template(
                        m, tokenize=True, add_generation_prompt=False,
                        enable_thinking=False))
                except TypeError:
                    ids.append(tokenizer.apply_chat_template(
                        m, tokenize=True, add_generation_prompt=False))
            enc = tokenizer.pad({"input_ids": ids}, padding=True,
                                return_tensors="pt").to(dev)
            logits = model(**enc).logits                     # (b, seq, vocab)
            last = enc["attention_mask"].sum(dim=1) - 1      # (b,)
            rows = torch.arange(logits.shape[0], device=dev)
            lt = logits[rows, last, TRUE_TOKEN]
            lf = logits[rows, last, FALSE_TOKEN]
            scores.append(torch.sigmoid(lt - lf).float().cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)


def ensure_reranker(model_name: str, device: str, model_dir: str = "") -> None:
    """经 model_manager 获取精排模型（常驻，touch 防卸载）。"""
    from core import model_manager
    mgr = model_manager.ModelManager.get()
    idle = _idle_seconds()
    mgr.slot("reranker", idle_seconds=idle).configure(
        lambda: load(model_name, device, model_dir), unload_fn=unload)
    if not is_loaded():
        mgr.acquire("reranker")
    else:
        mgr.touch("reranker")


def _idle_seconds() -> int:
    try:
        from core import config as cfg
        return int(cfg.load_config().get("idle_unload_seconds", 0))
    except Exception:
        return 0
