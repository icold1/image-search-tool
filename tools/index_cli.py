"""命令行索引/查询/caption 工具：不依赖 GUI 验证真实图片集。

用法:
    python tools/index_cli.py index <图片文件夹> [--batch 64]
    python tools/index_cli.py query "海边 日落" [--top 10]
    python tools/index_cli.py caption [--model Qwen/Qwen3-VL-2B-Instruct] [--retry-failed] [--force]
    python tools/index_cli.py embed [--dim 1024]
    python tools/index_cli.py stats
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config as cfg  # noqa: E402
from core import pipeline  # noqa: E402
from core import store as st  # noqa: E402


def get_store():
    return st.Store(str(cfg.DB_PATH), str(cfg.VEC_PATH),
                    cfg.MODEL_DIMS["ViT-L-14"])


def cmd_index(args):
    s = get_store()
    t0 = time.time()
    stats = pipeline.build_index(
        s, str(Path(args.folder).resolve()), "ViT-L-14", "auto", args.batch, 256,
        progress=lambda d, t, e: print(
            f"\r进度 {d}/{t} · 预计剩余 {e:.0f}s   ", end="", flush=True))
    total = s.count()  # 必须在 close 之前取
    s.close()
    print()
    print(f"完成: {stats}，耗时 {time.time() - t0:.0f}s，共 {total} 张")

    # 顺带把图片目录写入 GUI 配置，首次启动 GUI 无需再选
    conf = cfg.load_config()
    conf["image_dir"] = str(Path(args.folder).resolve())
    cfg.save_config(conf)
    print(f"已写入配置 image_dir = {conf['image_dir']}")


def cmd_query(args):
    s = get_store()
    t0 = time.time()
    res = pipeline.search(s, args.query, args.top, "ViT-L-14", "auto")
    ms = (time.time() - t0) * 1000
    print(f"查询「{args.query}」({ms:.0f}ms)，共 {len(res)} 条:")
    for i, (meta, score, tscore, desc) in enumerate(res, 1):
        tags = ""
        if tscore is not None:
            tags += f" [文:{tscore:.2f}]"
        if desc is not None:
            line = " ".join(str(desc.get("line", "")).split())[:40]
            tags += f" [述:{desc['score']:.2f}] {line}"
        print(f"  {i:3d}. {score:+.3f}{tags}  {Path(meta['path']).name}")
    s.close()


def cmd_caption(args):
    s = get_store()
    only = None
    if args.only:
        only = [line.lstrip("\ufeff").strip()
                for line in Path(args.only).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        print(f"仅处理清单中的 {len(only)} 个路径")
    t0 = time.time()
    stats = pipeline.build_captions(
        s, args.model, args.device, args.batch,
        include_failed=args.retry_failed, force=args.force, only_paths=only,
        progress=lambda d, t, e: print(
            f"\r描述进度 {d}/{t} · 预计剩余 {e:.0f}s   ", end="", flush=True),
        message=lambda m: print(f"\n{m}"))
    s.close()
    print()
    print(f"完成: {stats}，耗时 {time.time() - t0:.0f}s")
    print("提示：运行 `python tools/index_cli.py embed` 把描述构建为检索向量")


def cmd_embed(args):
    s = get_store()
    t0 = time.time()
    stats = pipeline.build_caption_vectors(
        s, args.model, args.dim, args.device,
        progress=lambda d, t, e: print(
            f"\r嵌入进度 {d}/{t}   ", end="", flush=True),
        message=lambda m: print(f"\n{m}"))
    s.close()
    print()
    print(f"完成: {stats}，耗时 {time.time() - t0:.0f}s")


def cmd_stats(_args):
    s = get_store()
    print(f"已索引图片: {s.count()} 张")
    print(f"向量矩阵: {s.vectors.shape} ({s.vectors.nbytes / 1024**2:.1f} MB)")
    print(f"墓碑行: {len(s.removed)}")
    print(f"待 OCR: {len(s.rows_missing_ocr())} 张")
    print(f"已有描述(caption): {s.count_captioned()} 张")
    print(f"待描述: {len(s.rows_missing_caption())} 张")
    print(f"描述失败: {s.count_caption_failed()} 张")
    print(f"caption 向量维度: {s.get_meta_value('caption_dim') or '未构建'}")
    print(f"caption 向量构建时间: {s.get_meta_value('caption_built_at') or '-'}")
    print(f"数据库: {cfg.DB_PATH}")
    print(f"向量文件: {cfg.VEC_PATH}")
    print(f"caption 向量: {cfg.CAPTION_VEC_PATH}")
    print(f"缩略图目录: {cfg.THUMB_DIR}")
    s.close()


def main():
    ap = argparse.ArgumentParser(description="图片语义索引 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_idx = sub.add_parser("index", help="增量建立索引")
    p_idx.add_argument("folder")
    p_idx.add_argument("--batch", type=int, default=64)
    p_qry = sub.add_parser("query", help="文本查询")
    p_qry.add_argument("query")
    p_qry.add_argument("--top", type=int, default=10)
    p_cap = sub.add_parser("caption", help="为图片生成结构化描述（本地 VLM）")
    p_cap.add_argument("--model", default=None,
                       help="VLM 名称（默认取 config 的 caption_model）")
    p_cap.add_argument("--device", default="auto")
    p_cap.add_argument("--batch", type=int, default=1)
    p_cap.add_argument("--only", default=None,
                       help="只处理该清单文件中的图片路径（每行一个，试跑验证用）")
    p_cap.add_argument("--retry-failed", action="store_true",
                       help="重试 status=failed 的图片")
    p_cap.add_argument("--force", action="store_true",
                       help="即使 config 关闭 caption 也执行")
    p_emb = sub.add_parser("embed", help="把描述构建为文本向量（检索第二路）")
    p_emb.add_argument("--model", default=None)
    p_emb.add_argument("--dim", type=int, default=None)
    p_emb.add_argument("--device", default="auto")
    sub.add_parser("stats", help="索引统计")
    args = ap.parse_args()
    {"index": cmd_index, "query": cmd_query, "caption": cmd_caption,
     "embed": cmd_embed, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
