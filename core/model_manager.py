"""模型生命周期管理：按名注册模型槽位，统一处理懒加载/空闲卸载/显存回收。

设计目标（与用户确认）：
- 查询速度优先：clip / embed 默认 idle_seconds=0（常驻不卸载）；
  config 的 idle_unload_seconds 可改为 >0 让它们空闲自动卸载。
- vlm（图像描述）只在 caption 批次期间需要，批次结束由调用方显式 unload。
- 所有加载/卸载线程安全；卸载伴随 gc + torch.cuda.empty_cache()。
"""
import gc
import threading
import time
from typing import Any, Callable, Dict, Optional


class ModelSlot:
    """单个模型的槽位。load_fn 返回模型对象（或 None 表示加载成功但无句柄）。"""

    def __init__(self, name: str, idle_seconds: int = 0):
        self.name = name
        self.idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self._obj: Any = None
        self._load_fn: Optional[Callable[[], Any]] = None
        self._unload_fn: Optional[Callable[[], None]] = None
        self._last_used = 0.0

    def configure(self, load_fn: Callable[[], Any],
                  unload_fn: Optional[Callable[[], None]] = None) -> None:
        with self._lock:
            self._load_fn = load_fn
            self._unload_fn = unload_fn

    def acquire(self, load_fn: Optional[Callable[[], Any]] = None) -> Any:
        """取模型对象；未加载则同步加载。返回模型对象（可能为 None 占位）。"""
        with self._lock:
            if load_fn is not None and self._load_fn is None:
                self._load_fn = load_fn
            if self._obj is None:
                if self._load_fn is None:
                    raise RuntimeError(f"模型槽位 {self.name} 未配置加载函数")
                self._obj = self._load_fn()
                if self._obj is None:
                    self._obj = True  # 加载成功但无需持有句柄
            self._last_used = time.monotonic()
            return None if self._obj is True else self._obj

    def touch(self) -> None:
        with self._lock:
            self._last_used = time.monotonic()

    def is_loaded(self) -> bool:
        with self._lock:
            return self._obj is not None

    def unload(self) -> None:
        with self._lock:
            obj, self._obj = self._obj, None
        if obj is None:
            return
        try:
            if self._unload_fn is not None:
                self._unload_fn()
        finally:
            self._release_gpu()

    def maybe_idle_unload(self) -> None:
        if self.idle_seconds <= 0:
            return
        with self._lock:
            if self._obj is None:
                return
            idle = time.monotonic() - self._last_used
        if idle >= self.idle_seconds:
            self.unload()

    @staticmethod
    def _release_gpu() -> None:
        try:
            gc.collect()
            import torch  # 延迟导入，未装 torch 的环境不受影响
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _class_lock = threading.Lock()

    @classmethod
    def get(cls) -> "ModelManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._slots: Dict[str, ModelSlot] = {}
        self._lock = threading.Lock()
        self._reaper = threading.Thread(target=self._loop, daemon=True,
                                        name="model-reaper")
        self._reaper.start()

    def slot(self, name: str, idle_seconds: int = 0) -> ModelSlot:
        with self._lock:
            s = self._slots.get(name)
            if s is None:
                s = ModelSlot(name, idle_seconds)
                self._slots[name] = s
            else:
                s.idle_seconds = idle_seconds  # 允许后续调整空闲策略
            return s

    def acquire(self, name: str, load_fn: Optional[Callable[[], Any]] = None) -> Any:
        return self.slot(name).acquire(load_fn)

    def touch(self, name: str) -> None:
        with self._lock:
            s = self._slots.get(name)
        if s is not None:
            s.touch()

    def unload(self, name: str) -> None:
        with self._lock:
            s = self._slots.get(name)
        if s is not None:
            s.unload()

    def unload_all(self, keep=()) -> None:
        with self._lock:
            names = [n for n in self._slots if n not in keep]
        for n in names:
            self._slots[n].unload()

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            s = self._slots.get(name)
        return bool(s) and s.is_loaded()

    def _loop(self) -> None:
        while True:
            time.sleep(1)
            with self._lock:
                slots = list(self._slots.values())
            for s in slots:
                s.maybe_idle_unload()
