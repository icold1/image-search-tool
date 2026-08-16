"""VLM caption A/B 评测：对比不同本地视觉模型的 质量样例/速度/显存。

用途（第二期选型验收，8GB 显存机器）：
    python tools/vlm_bench.py --images D:/path/to/sample_dir --limit 30 \
        --models Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen2.5-VL-7B-Instruct \
        --out data/vlm_bench

输出：每个模型一份 <模型名>.json（含每张图的 caption 与耗时）+ 控制台汇总
（平均秒/张、吞吐、峰值显存），以及每模型 2 条 caption 样例供人工比对质量。

注意：
- Qwen2.5-VL-7B-Instruct fp16 需 ~16GB 显存；8GB 卡请自行准备 INT4 量化权重
  （如 GPTQ/AWQ，装 auto-gptq），把 --models 换成本地量化权重目录路径。
- 首次运行会从 ModelScope 下载权重（失败回退 HuggingFace）。
"""
import argparse
import json
import sys
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


def bench_one(model_name: str, files: list, device: str, out_dir: Path,
              max_new_tokens: int):
    from core._win import ensure_env_dlls
    ensure_env_dlls()  # 未激活 conda 环境直跑时也能找到 cuDNN
    import torch
    from core import captioner
    from core import thumbnailer

    print(f"\n===== 模型: {model_name} =====")
    cap = captioner.LocalVlmCaptioner(model_name, device,
                                      max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = []
    t0 = time.time()
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
        print(f"\r  {len(results)}/{len(files)}  {results[-1]['file'][:40]:40s}"
              f"  {results[-1]['seconds']:.1f}s   ", end="", flush=True)
    total = time.time() - t0
    cap.unload()
    print()
    ok = [r for r in results if "caption" in r]
    avg = (sum(r["seconds"] for r in results) / len(results)) if results else 0
    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
               if torch.cuda.is_available() else 0.0)
    summary = {
        "model": model_name,
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
                    help="模型名/本地路径，可多个")
    ap.add_argument("--limit", type=int, default=0, help="每模型最多评测张数")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(Path(ROOT) / "data" / "vlm_bench"))
    ap.add_argument("--max-new-tokens", type=int, default=600)
    args = ap.parse_args()

    files = collect_images(args.images, args.limit)
    if not files:
        raise SystemExit("未找到任何图片")
    print(f"评测图片 {len(files)} 张，模型 {len(args.models)} 个")

    summaries = []
    for m in args.models:
        summaries.append(bench_one(m, files, args.device, Path(args.out),
                                   args.max_new_tokens))
    print("\n===== 汇总对比 =====")
    for s in summaries:
        print(f"{s['model']}: {s['avg_seconds_per_image']}s/张 · "
              f"{s['images_per_minute']} 张/分 · 峰值显存 {s['peak_vram_mb']}MB"
              f" · 失败 {s['failed']}/{s['images']}")
    print("\n请人工比对各模型 JSON 中的 caption 样例质量后再决定全量用哪个。")


if __name__ == "__main__":
    main()
