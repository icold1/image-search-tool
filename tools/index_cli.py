"""命令行索引/查询工具：不依赖 GUI 验证真实图片集，也可日常重建索引。

用法:
    python tools/index_cli.py index <图片文件夹> [--batch 64]
    python tools/index_cli.py query "海边 日落" [--top 10]
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
    for i, (meta, score, tscore) in enumerate(res, 1):
        tag = f" [文:{tscore:.2f}]" if tscore is not None else ""
        print(f"  {i:3d}. {score:+.3f}{tag}  {Path(meta['path']).name}")
    s.close()


def cmd_stats(_args):
    s = get_store()
    print(f"已索引图片: {s.count()} 张")
    print(f"向量矩阵: {s.vectors.shape} ({s.vectors.nbytes / 1024**2:.1f} MB)")
    print(f"墓碑行: {len(s.removed)}")
    print(f"待 OCR: {len(s.rows_missing_ocr())} 张")
    print(f"数据库: {cfg.DB_PATH}")
    print(f"向量文件: {cfg.VEC_PATH}")
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
    sub.add_parser("stats", help="索引统计")
    args = ap.parse_args()
    {"index": cmd_index, "query": cmd_query, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
