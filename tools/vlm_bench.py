"""VLM caption A/B 评测：对比不同本地视觉模型的 质量样例/速度/显存。

用途（8GB 显存机器）：
    python tools/vlm_bench.py --images D:/path/to/sample_dir --limit 30 \
        --models Qwen/Qwen3-VL-2B-Instruct --out data/vlm_bench

GGUF/llama.cpp 后端（模型路径以 .gguf 结尾）：
    python tools/vlm_bench.py --images data/vlm_bench/sample_list.txt --limit 30 \
        --models data/models/cache/gguf/Qwen3VL-2B-Instruct-Q8_0.gguf \
        --mmproj data/models/cache/gguf/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
        --llama-bin data/deps/llama-b10453-cuda-12.4

输出：每个模型一份 <模型名>.json（含每张图的 caption 与耗时）+ 控制台汇总
（平均秒/张、吞吐、峰值显存），以及每模型 2 条 caption 样例供人工比对质量。

注意：transformers 后端首次运行会从 ModelScope 下载权重（失败回退 HF）；
GGUF 后端使用 llama.cpp 官方预编译 llama-server（独立进程，显存经 nvidia-smi 采样）。
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(images_arg: str, limit: int) -> list:
    p = Path(images_arg)
    if p.is_file():
        files = [line.lstrip("\ufeff").strip()
                 for line in p.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    elif p.is_dir():
        files = sorted(str(f) for f in p.rglob("*")
                       if f.suffix.lower() in IMAGE_EXTS)
    else:
        raise SystemExit(f"路径不存在: {images_arg}")
    return files[:limit] if limit else files


class VramSampler:
    """nvidia-smi 采样线程：记录评测期间的显存峰值（GGUF 后端用）。"""

    def __init__(self, interval: float = 2.0):
        self._interval = interval
        self._peak = 0.0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._th.start()

    def stop(self) -> float:
        self._stop.set()
        self._th.join(timeout=5)
        return self._peak

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                used = float(out.stdout.strip().splitlines()[0])
                self._peak = max(self._peak, used)
            except Exception:
                pass
            self._stop.wait(self._interval)


def bench_one(model_name: str, files: list, device: str, out_dir: Path,
              max_new_tokens: int, mmproj: str = "", llama_bin: str = ""):
    from core._win import ensure_env_dlls
    ensure_env_dlls()  # 未激活 conda 环境直跑时也能找到 cuDNN
    import torch
    from core import captioner
    from core import thumbnailer

    is_gguf = model_name.lower().endswith(".gguf")
    print(f"\n===== 模型: {model_name} "
          f"({'llama.cpp/GGUF' if is_gguf else 'transformers/fp16'}) =====")
    sampler = None
    if is_gguf:
        cap = captioner.LlamaCppCaptioner(model_name, mmproj_path=mmproj or None,
                                          bin_dir=llama_bin or None,
                                          max_new_tokens=max_new_tokens)
        sampler = VramSampler()
        sampler.start()
    else:
        cap = captioner.LocalVlmCaptioner(model_name, device,
                                          max_new_tokens=max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    results = []
    t0 = time.time()
    try:
        for f in files:
            t_img = time.time()
            try:
                img = thumbnailer.load_image(f)
                data = cap.describe(img)
                results.append({"file": Path(f).name, "caption": data,
                                "seconds": round(time.time() - t_img, 2)})
            except Exception as e:
                results.append({"file": Path(f).name, "error": str(e)[:200],
                                "seconds": round(time.time() - t_img, 2)})
            print(f"\r  {len(results)}/{len(files)}  "
                  f"{results[-1]['file'][:40]:40s}"
                  f"  {results[-1]['seconds']:.1f}s   ", end="", flush=True)
    finally:
        cap.unload()
    total = time.time() - t0
    print()
    ok = [r for r in results if "caption" in r]
    avg = (sum(r["seconds"] for r in results) / len(results)) if results else 0
    if is_gguf:
        peak_mb = sampler.stop() if sampler else 0.0
    else:
        peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
                   if torch.cuda.is_available() else 0.0)
    summary = {
        "model": model_name,
        "backend": "llama.cpp" if is_gguf else "transformers",
        "images": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "total_seconds": round(total, 1),
        "avg_seconds_per_image": round(avg, 2),
        "images_per_minute": round(len(results) / (total / 60), 1) if total else 0,
        "peak_vram_mb": round(peak_mb, 0),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace("/", "--").replace("\\", "--")
    with open(out_dir / f"{slug}.json", "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "results": results}, fp,
                  ensure_ascii=False, indent=2)
    print(f"  汇总: {summary}")
    print(f"  结果保存: {out_dir / (slug + '.json')}")
    for r in ok[:2]:
        cap_text = json.dumps(r["caption"], ensure_ascii=False)[:400]
        print(f"  ---- 样例 {r['file']} ----\n  {cap_text}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="VLM caption A/B 评测")
    ap.add_argument("--images", required=True,
                    help="图片目录或图片路径列表文件")
    ap.add_argument("--models", nargs="+", required=True,
                    help="模型名/本地路径（.gguf 结尾走 llama.cpp 后端），可多个")
    ap.add_argument("--limit", type=int, default=0, help="每模型最多评测张数")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(Path(ROOT) / "data" / "vlm_bench"))
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--mmproj", default="", help="GGUF 模型的视觉投影文件")
    ap.add_argument("--llama-bin", default="",
                    help="llama-server.exe 所在目录（空=自动查找）")
    args = ap.parse_args()

    files = collect_images(args.images, args.limit)
    if not files:
        raise SystemExit("未找到任何图片")
    print(f"评测图片 {len(files)} 张，模型 {len(args.models)} 个")

    summaries = []
    for m in args.models:
        summaries.append(bench_one(m, files, args.device, Path(args.out),
                                   args.max_new_tokens,
                                   mmproj=args.mmproj,
                                   llama_bin=args.llama_bin))
    print("\n===== 汇总对比 =====")
    for s in summaries:
        print(f"{s['model']} [{s['backend']}]: {s['avg_seconds_per_image']}s/张 · "
              f"{s['images_per_minute']} 张/分 · 峰值显存 {s['peak_vram_mb']}MB"
              f" · 失败 {s['failed']}/{s['images']}")
    print("\n请人工比对各模型 JSON 中的 caption 样例质量后再决定全量用哪个。")


if __name__ == "__main__":
    main()
