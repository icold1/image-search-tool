"""llama-server 本地服务管理：llama.cpp 官方预编译二进制 + OpenAI 兼容接口。

Windows 上 llama-cpp-python 无官方 CUDA 轮子，改用 llama.cpp 官方发布的
预编译 CUDA 二进制（llama-server.exe），以子进程方式启动本地 HTTP 服务，
通过 /v1/chat/completions（OpenAI 兼容）做多模态对话。项目自包含、无需编译。

注意：子进程 stdout/stderr 重定向到日志文件（沙箱/Windows 下管道捕获受限）。
"""
import base64
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_PORT = 18080


class LlamaServerError(Exception):
    """llama-server 启动/调用失败。"""


def _find_server_exe(bin_dir: Optional[str]) -> str:
    candidates = []
    if bin_dir:
        candidates.append(Path(bin_dir) / "llama-server.exe")
    root = Path(__file__).resolve().parent.parent
    candidates += [
        root / "data" / "deps" / "llama-server.exe",          # 解压后直接在此
        root / "data" / "deps" / "llama-b10453-cuda-12.4" / "llama-server.exe",
        root / "data" / "deps" / "llama-b10453-cuda-13.3" / "llama-server.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise LlamaServerError(
        "未找到 llama-server.exe：请把 llama.cpp 的 Windows CUDA 预编译包解压到 "
        "data/deps/ 下，或通过 config 的 caption_llama_bin 指定目录")


class LlamaServer:
    """管理一个 llama-server 进程（模型 + mmproj 视觉投影）。"""

    def __init__(self, model_path: str, mmproj_path: Optional[str] = None,
                 bin_dir: Optional[str] = None, port: int = DEFAULT_PORT,
                 n_gpu_layers: int = 999, ctx_size: int = 8192,
                 log_path: Optional[str] = None):
        self.model_path = str(Path(model_path).resolve())
        self.mmproj_path = str(Path(mmproj_path).resolve()) if mmproj_path else None
        self.bin_dir = bin_dir
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.log_path = log_path or str(
            Path(__file__).resolve().parent.parent / "data" / "logs"
            / "llama-server.log")
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    # ---------- 生命周期 ----------
    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            exe = _find_server_exe(self.bin_dir)
            cmd = [exe, "-m", self.model_path,
                   "--host", "127.0.0.1", "--port", str(self.port),
                   "-ngl", str(self.n_gpu_layers), "-c", str(self.ctx_size),
                   "--parallel", "1", "--no-webui"]
            if self.mmproj_path:
                cmd += ["--mmproj", self.mmproj_path]
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            logf = open(self.log_path, "a", encoding="utf-8")
            logf.write(f"\n===== start {time.strftime('%H:%M:%S')} "
                       f"{Path(self.model_path).name} =====\n")
            logf.flush()
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, creationflags=flags)
            self._wait_ready()

    def _wait_ready(self, timeout: float = 300.0) -> None:
        deadline = time.monotonic() + timeout
        last_err = ""
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                raise LlamaServerError(
                    f"llama-server 进程退出（code={self._proc.poll()}），"
                    f"详见 {self.log_path}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health",
                        timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5)
        raise LlamaServerError(f"llama-server 启动超时：{last_err}")

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ---------- 对话 ----------
    def chat(self, image_b64: Optional[str], prompt: str,
             max_tokens: int = 600) -> str:
        """单轮多模态对话，返回文本。失败抛 LlamaServerError。

        采样参数说明：量化小模型 greedy 解码易陷入重复循环（实测 2B-Q8
        在长 JSON 生成时重复同一 object 直到截断），因此用温和采样
        （temperature 0.2 + top_p 0.9）+ repeat_penalty + JSON 语法约束。
        """
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })
        payload = {
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
        }
        try:
            resp = self._post("/v1/chat/completions", payload)
        except LlamaServerError:
            # 老版本/不支持 response_format：去掉后按纯采样重试
            payload.pop("response_format", None)
            resp = self._post("/v1/chat/completions", payload)
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LlamaServerError(f"响应格式异常: {resp}") from e

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise LlamaServerError(f"HTTP {e.code}: {body}") from e
        except Exception as e:
            raise LlamaServerError(f"请求失败: {e}") from e
