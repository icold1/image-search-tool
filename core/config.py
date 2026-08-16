"""全局配置：默认值 + config.json 持久化。"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
THUMB_DIR = DATA_DIR / "thumbs"
MODEL_DIR = DATA_DIR / "models" / "checkpoints"
DB_PATH = DATA_DIR / "index.db"
VEC_PATH = DATA_DIR / "vectors.npy"
CAPTION_VEC_PATH = DATA_DIR / "caption_vecs.npy"   # caption 行向量矩阵
CAPTION_AGG_PATH = DATA_DIR / "caption_agg.npy"    # 每图 max-pool 聚合向量
CONFIG_PATH = DATA_DIR / "config.json"

# cn_clip 模型名 -> 特征维度
MODEL_DIMS = {
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-L-14-336": 768,
    "RN50": 1024,
}

DEFAULT_CONFIG = {
    "image_dir": "",            # 图片文件夹（首次启动时选择）
    "top_k": 10,                # 默认显示数量，悬浮球菜单可调
    "model_name": "ViT-L-14",   # Chinese-CLIP 模型
    "device": "auto",           # auto / cuda / cpu
    "batch_size": 64,           # 建索引时的批大小
    "thumb_size": 256,          # 缩略图最长边（像素）
    "hotkey": "Ctrl+Shift+Space",
    "ocr_enabled": True,        # 是否对图片做 OCR 文字识别（文字优先匹配）
    "ball_pos": [None, None],   # 悬浮球上次位置 [x, y]
    "extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"],
    # ---------- caption（图像描述）----------
    "caption_enabled": True,    # 是否启用 caption 路（无 GPU 自动降级）
    "caption_model": "Qwen/Qwen3-VL-2B-Instruct",  # 本地 VLM（默认 8GB 卡档位）
    "caption_device": "auto",   # caption 用 auto / cuda / cpu
    "caption_batch": 1,         # VLM 批大小（8GB 卡保持 1）
    "caption_max_new_tokens": 600,  # 单图生成上限（JSON 描述无需更长）
    "caption_keep_clip": True,  # caption 期间保留 CLIP 常驻（搜索可用）
    # ---------- GGUF/llama.cpp 后端（caption_model 以 .gguf 结尾时启用）----------
    "caption_mmproj": "",       # mmproj 视觉投影文件路径
    "caption_llama_bin": "",    # llama-server.exe 所在目录（空=自动查找 data/deps）
    # ---------- caption 文本向量 ----------
    "embed_model": "Qwen/Qwen3-Embedding-0.6B",
    "embed_dim": 1024,          # MRL 截断维度（1024 全量，可降到 512 省内存）
    "caption_search_enabled": True,  # 搜索时是否使用 caption 向量路
    # ---------- 模型生命周期 ----------
    "idle_unload_seconds": 0,   # 0 = CLIP/嵌入模型常驻（查询速度优先）
    # ---------- 融合 ----------
    "fusion": {
        "w_clip": 1.0,          # CLIP 全局语义权重
        "w_caption": 0.6,        # caption 文本向量权重
        "rrf_k": 60,             # RRF 常数
        "candidate_n": 100,      # 每路取 top-N 参与融合
    },
    # ---------- 精排（交叉编码，可选）----------
    "rerank_enabled": False,     # 默认关闭：首次启用需下载模型，建议准备好后手动开启
    "rerank_model": "Qwen/Qwen3-Reranker-0.6B",
    "rerank_top_n": 30,          # 对融合结果前 N 张做精排
    "rerank_batch": 16,
    # ---------- API（仅手动触发）----------
    "api_manual_only": True,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    # 嵌套 dict 的默认值逐键补全（旧配置可能缺子键）
    if isinstance(cfg.get("fusion"), dict):
        for k, v in DEFAULT_CONFIG["fusion"].items():
            cfg["fusion"].setdefault(k, v)
    else:
        cfg["fusion"] = dict(DEFAULT_CONFIG["fusion"])
    return cfg


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
