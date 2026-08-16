"""端到端冒烟测试（无需 GUI）：生成测试图 -> 建索引 -> 查询 -> 验证。

用法（在 image_caption_and_select 环境）:
    python tools/smoke_test.py

首次运行会下载 Chinese-CLIP ViT-L-14 权重（约 1.6GB）。
"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from core import config as cfg  # noqa: E402
from core import model as ml  # noqa: E402
from core import pipeline  # noqa: E402
from core import store as st  # noqa: E402

SMOKE_ROOT = cfg.DATA_DIR / "smoke"
IMG_DIR = SMOKE_ROOT / "images"
SMOKE_DB = SMOKE_ROOT / "index.db"
SMOKE_VEC = SMOKE_ROOT / "vectors.npy"
SMOKE_THUMBS = SMOKE_ROOT / "thumbs"


def make_test_images(folder: Path) -> int:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    specs = [
        ("01_blue_sky.png", "#87CEEB"),
        ("02_orange_sunset.png", "#FF7E5F"),
        ("03_green_forest.png", "#2E7D32"),
        ("04_red_flower.png", "#D0021B"),
        ("05_white_snow.png", "#F5F5F5"),
        ("06_yellow_desert.png", "#F2C94C"),
    ]
    for name, color in specs:
        img = Image.new("RGB", (320, 240), color)
        d = ImageDraw.Draw(img)
        d.ellipse((110, 70, 210, 170), fill="white")
        d.rectangle((20, 200, 300, 240), fill=(60, 60, 60))
        img.save(folder / name)
    return len(specs)


def main() -> int:
    shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
    n = make_test_images(IMG_DIR)
    print(f"[1/4] 生成测试图片 {n} 张 -> {IMG_DIR}")

    print("[2/4] 加载模型（首次需下载权重，请耐心等待）...")
    t0 = time.time()
    ml.load_model("ViT-L-14", "auto", str(cfg.MODEL_DIR))
    dim = ml.feature_dim()
    print(f"      模型加载完成，特征维度={dim}，用时 {time.time() - t0:.1f}s")

    print("[3/4] 建立索引...")
    s = st.Store(str(SMOKE_DB), str(SMOKE_VEC), dim)
    t0 = time.time()
    stats = pipeline.build_index(
        s, str(IMG_DIR), "ViT-L-14", "auto", batch_size=8, thumb_size=256,
        thumbs_dir=str(SMOKE_THUMBS),
        progress=lambda d, t, e: print(f"      进度 {d}/{t}"))
    print(f"      索引完成: {stats}，耗时 {time.time() - t0:.1f}s")
    assert s.count() == n, f"索引数量不符: {s.count()} != {n}"

    print("[4/4] 查询验证...")
    for q in ["天空", "日落", "森林", "红色 花"]:
        t0 = time.time()
        res = pipeline.search(s, q, 3)
        ms = (time.time() - t0) * 1000
        names = [Path(m["path"]).name for m, _, _ in res]
        scores = [f"{sc:.3f}" for _, sc, _ in res]
        print(f"      查询「{q}」-> {list(zip(names, scores))} ({ms:.0f}ms)")
    assert pipeline.search(s, "天空", 3), "查询无结果"
    s.close()

    print("\n[OK] 冒烟测试通过：核心链路（模型加载/索引/检索）正常。")
    print("注：测试图为合成色块，语义结果无参考意义；请用真实图片验证效果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
