"""QThread 包装：把管线任务放到后台线程，避免阻塞 UI。"""
import threading
import time
import traceback

from PySide6.QtCore import QThread, Signal

from core import pipeline


class IndexWorker(QThread):
    progress = Signal(int, int, float)   # done, total, eta_seconds
    message = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, store, image_dir, model_name, device, batch_size,
                 thumb_size, parent=None):
        super().__init__(parent)
        self._store = store
        self._image_dir = image_dir
        self._model_name = model_name
        self._device = device
        self._batch = batch_size
        self._thumb = thumb_size
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            stats = pipeline.build_index(
                self._store, self._image_dir, self._model_name, self._device,
                self._batch, self._thumb,
                progress=lambda d, t, e: self.progress.emit(d, t, e),
                message=lambda m: self.message.emit(m),
                stop=self._stop)
            self.finished_ok.emit(stats)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class SearchWorker(QThread):
    """查询线程：version 用于丢弃过期结果；cancel() 打断在途查询。"""
    results_ready = Signal(list, float, int)   # [(meta, score, ...)], elapsed_ms, version
    failed = Signal(str, int)                  # msg, version

    def __init__(self, store, query, k, model_name, device, version=0,
                 parent=None):
        super().__init__(parent)
        self._store = store
        self._query = query
        self._k = k
        self._model_name = model_name
        self._device = device
        self._version = version
        self._stop = threading.Event()

    def cancel(self):
        self._stop.set()

    def run(self):
        t0 = time.time()
        try:
            res = pipeline.search(self._store, self._query, self._k,
                                  self._model_name, self._device,
                                  stop=self._stop)
            self.results_ready.emit(res, (time.time() - t0) * 1000,
                                    self._version)
        except pipeline.SearchCancelled:
            return  # 被取消：静默丢弃
        except Exception as e:
            self.failed.emit(traceback.format_exc(), self._version)


class CaptionWorker(QThread):
    """后台图像描述（caption）任务：可暂停、逐图失败隔离。"""
    progress = Signal(int, int, float)
    message = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, store, model_name, device, batch_size,
                 include_failed=False, parent=None):
        super().__init__(parent)
        self._store = store
        self._model_name = model_name
        self._device = device
        self._batch = batch_size
        self._include_failed = include_failed
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            stats = pipeline.build_captions(
                self._store, self._model_name, self._device, self._batch,
                include_failed=self._include_failed,
                progress=lambda d, t, e: self.progress.emit(d, t, e),
                message=lambda m: self.message.emit(m),
                stop=self._stop)
            self.finished_ok.emit(stats)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class EmbedWorker(QThread):
    """后台 caption 文本向量构建任务。"""
    progress = Signal(int, int, float)
    message = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, store, embed_model, dim, device, parent=None):
        super().__init__(parent)
        self._store = store
        self._embed_model = embed_model
        self._dim = dim
        self._device = device
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            stats = pipeline.build_caption_vectors(
                self._store, self._embed_model, self._dim, self._device,
                progress=lambda d, t, e: self.progress.emit(d, t, e),
                message=lambda m: self.message.emit(m),
                stop=self._stop)
            self.finished_ok.emit(stats)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")
