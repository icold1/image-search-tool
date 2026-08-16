"""Qt6Core.dll 加载实验：按参数组合搜索标志，定位冲突 DLL 来源。

用法: python tools/qt_load_test.py <test>
  test = A: DLL_LOAD_DIR|SYSTEM32
         B: A + DEFAULT_DIRS
         C: A + USER_DIRS
         D: DEFAULT_DIRS|DLL_LOAD_DIR + 清空 PATH（仅 System32）
         E: DEFAULT_DIRS|DLL_LOAD_DIR + 原 PATH
"""
import ctypes
import os
import sys

QT = (r"D:\Data\Anaconda\envs\image_caption_and_select"
      r"\Lib\site-packages\PySide6\Qt6Core.dll")

DLD = 0x100   # LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
SYS32 = 0x800
DEFDIRS = 0x1000
USERDIRS = 0x400

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

test = sys.argv[1] if len(sys.argv) > 1 else "A"

if test == "D":
    os.environ["PATH"] = r"C:\Windows\System32"

flags = {
    "A": DLD | SYS32,
    "B": DLD | SYS32 | DEFDIRS,
    "C": DLD | SYS32 | USERDIRS,
    "D": DLD | DEFDIRS,
    "E": DLD | DEFDIRS,
}[test]

h = kernel32.LoadLibraryExW(QT, None, flags)
if h:
    print(f"[{test}] OK  handle={hex(h)}")
else:
    print(f"[{test}] FAIL err={ctypes.get_last_error()}")
