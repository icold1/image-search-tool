"""Windows 启动引导：注册 conda 环境的 DLL 目录。

未激活 conda 环境而直接运行环境内 python 时，env\\Library\\bin 不在
DLL 搜索路径中，torch 会找不到 cuDNN 等动态库。两种手段并用：

1. os.add_dll_directory —— 对使用 USER_DIRS 搜索的加载器有效；
2. 把目录前插到 PATH —— torch 的 cudnn 加载器按默认搜索目录
   （含 PATH）查找，必须在 import torch 之前调用。
"""
import os
import sys
from pathlib import Path


def ensure_env_dlls() -> None:
    if os.name != "nt":
        return
    candidates = []
    try:
        candidates.append(Path(sys.prefix) / "Library" / "bin")
    except Exception:
        pass
    # venv 场景：sys.prefix 是 .venv，真实环境在 sys.base_prefix
    try:
        if sys.base_prefix != sys.prefix:
            candidates.append(Path(sys.base_prefix) / "Library" / "bin")
    except Exception:
        pass
    try:
        candidates.append(
            Path(sys.executable).resolve().parent.parent / "Library" / "bin")
    except Exception:
        pass
    seen = set()
    for d in candidates:
        try:
            if not d.is_dir():
                continue
            s = str(d)
            if s in seen:
                continue
            seen.add(s)
            try:
                os.add_dll_directory(s)
            except Exception:
                pass
            path = os.environ.get("PATH", "")
            if s not in path.split(os.pathsep):
                os.environ["PATH"] = s + os.pathsep + path
        except Exception:
            pass
