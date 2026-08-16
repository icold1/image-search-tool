"""缩略图生成与缓存。

缩略图文件名用"源路径的 md5"作为 key，跨索引重建保持稳定，
避免重复生成。原图文件夹只读，不产生任何改动。
"""
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

MAX_PIXELS = 60_000_000  # 加载超大图时先降采样，防止内存爆炸


def thumb_key(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


def thumb_path_for(thumbs_dir: str, image_path: str) -> str:
    return str(Path(thumbs_dir) / f"{thumb_key(image_path)}.jpg")


def load_image(image_path: str, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """安全加载图片：EXIF 方向纠正、超大图降采样、动图取首帧。

    抛异常时由调用方记入 skipped.log。
    """
    img = Image.open(image_path)
    img.load()
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    if img.width * img.height > max_pixels:
        scale = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS)
    return img


def save_thumbnail(img: Image.Image, thumbs_dir: str, image_path: str,
                   size: int = 256, force: bool = False) -> str:
    """从已加载的 PIL 图生成缩略图，返回缩略图路径；已存在则跳过。

    force=True 时无条件重建（图片内容已变更、旧缩略图过期时使用）。
    """
    out = thumb_path_for(thumbs_dir, image_path)
    if force and Path(out).exists():
        Path(out).unlink()
    if not Path(out).exists():
        Path(thumbs_dir).mkdir(parents=True, exist_ok=True)
        im = img.copy()
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        im.convert("RGB").save(out, "JPEG", quality=82)
    return out


def make_thumbnail(image_path: str, thumbs_dir: str, size: int = 256,
                   force: bool = False) -> str:
    """按路径生成缩略图（会重新打开图片），返回缩略图路径。"""
    return save_thumbnail(load_image(image_path), thumbs_dir, image_path, size,
                          force=force)
