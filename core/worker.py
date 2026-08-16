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
    results_ready = Signal(list, float)  # [(meta, score)], elapsed_ms
    failed = Signal(str)

    def __init__(self, store, query, k, model_name, device, parent=None):
        super().__init__(parent)
        self._store = store
        self._query = query
        self._k = k
        self._model_name = model_name
        self._device = device

    def run(self):
        t0 = time.time()
        try:
            res = pipeline.search(self._store, self._query, self._k,
                                  self._model_name, self._device)
            self.results_ready.emit(res, (time.time() - t0) * 1000)
        except Exception as e:
            self.failed.emit(traceback.format_exc())
