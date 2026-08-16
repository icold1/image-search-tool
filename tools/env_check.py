"""环境探测脚本：检查 Python 版本、关键依赖与 GPU 状态。

用法: python tools/env_check.py
"""
import importlib
import sys

print("Python:", sys.version)

mods = ["torch", "torchvision", "cn_clip", "PySide6", "PIL", "numpy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "ok")
        print(f"{m:12s} -> {ver}")
    except Exception as e:
        print(f"{m:12s} -> MISSING ({type(e).__name__}: {e})")

try:
    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("torch built cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("VRAM (GB):", round(total_gb, 1))
except Exception as e:
    print("torch probe failed:", e)
