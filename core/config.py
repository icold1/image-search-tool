"""全局配置：默认值 + config.json 持久化。"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
THUMB_DIR = DATA_DIR / "thumbs"
MODEL_DIR = DATA_DIR / "models" / "checkpoints"
DB_PATH = DATA_DIR / "index.db"
VEC_PATH = DATA_DIR / "vectors.npy"
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
    "ocr_enabled": True,       # 是否对图片做 OCR 文字识别（文字优先匹配）
    "ball_pos": [None, None],   # 悬浮球上次位置 [x, y]
    "extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"],
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
