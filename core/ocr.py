"""OCR 文字识别：RapidOCR（PP-OCRv4 模型，CPU 推理，中文友好）。

引擎为懒加载单例（首次初始化约 1~3 秒），线程安全。
"""
import threading

import numpy as np

_lock = threading.Lock()
_engine = None

MAX_TEXT_LEN = 4000


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        return _engine


def extract_text(img) -> str:
    """PIL 图像 -> 识别出的文字（多行以换行分隔，截断到 4000 字符）。

    识别失败返回空字符串。
    """
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR（OpenCV 格式）
    try:
        result, _ = _get_engine()(arr)
    except Exception:
        return ""
    if not result:
        return ""
    lines = []
    for item in result:
        try:
            text = str(item[1]).strip()
            if text:
                lines.append(text)
        except Exception:
            continue
    return "\n".join(lines)[:MAX_TEXT_LEN]
