# 图片语义搜索工具（本地 Chinese-CLIP + PySide6 悬浮球）

一个完全本地运行的图片语义搜索工具：屏幕上方常驻悬浮球，点击（或
`Ctrl+Shift+空格`）弹出查询框，输入中文描述（停止输入 1 秒自动搜索），
显示最匹配的图片缩略图，可直接**拖拽到微信聊天**发送。

## 特性

- **完全本地**：Chinese-CLIP ViT-L/14 语义检索 + PP-OCRv4 文字识别，无任何云端 API
- **中文自然语言搜图**：支持空格分隔多关键词（如"海边 日落"，取词向量平均）
- **图片文字优先匹配**：后台 OCR 识别图中文字建立倒排索引，查询词命中图中
  文字的结果排最前（角标"文"），其余按语义相似度补足；悬停可见命中的文字
- **悬浮球**：置顶、可拖动、记住位置；右键菜单：刷新索引 / 显示数量（5/10/20/50）/ 退出
- **查询面板跟随悬浮球**：水平居中于球下方（空间不足自动翻到上方，拖动球时
  实时跟随）；**无搜索时收缩为仅输入框**，出结果才展开图片区
- **系统托盘**：右键可显示/隐藏悬浮球、打开查询框、刷新索引、设置显示数量、
  查看索引数量、退出；单击/双击托盘图标呼出查询框
- **结果交互**：单击缩略图打开原图；**按住拖到微信直接发送文件**；右键复制
  文件 / 打开所在文件夹 / 从索引移除；滚轮在面板任意位置均可滚动结果
- **增量索引**："刷新索引"按 mtime 只处理新增/改动图片，中途退出自动续传；
  OCR 同样增量（仅识别未处理过的图，首次全量约 0.5 秒/张，CPU，可边识别边搜索）
- **原图文件夹只读**：只生成缩略图缓存与向量库，绝不改动原图

## 使用速查

| 操作 | 方式 |
|---|---|
| 打开/收起查询框 | 单击悬浮球 / `Ctrl+Shift+空格` / 单击托盘图标 |
| 搜索 | 输入中文描述，停止 1 秒自动搜索（回车立即搜索） |
| 多关键词 | 空格分隔，如"海边 日落" |
| 发送到微信 | 按住缩略图拖到聊天窗（备选：右键"复制文件"→ Ctrl+V） |
| 打开原图 / 定位文件 | 单击缩略图 / 右键"打开所在文件夹" |
| 刷新索引 | 悬浮球或托盘右键 → 刷新索引 |
| 隐藏悬浮球 | 托盘右键 → 取消勾选"显示悬浮球" |
| 退出 | 悬浮球/托盘右键 → 退出 |

## 环境要求

- Windows 11（当前为单机使用设计，热键/拖拽基于 Windows）
- NVIDIA GPU（RTX 50 系需 CUDA 12.8+ / PyTorch 2.8+ cu128；无 GPU 也能跑，
  仅建索引变慢）；16GB 内存可支撑 10 万张图片
- conda 环境：`image_caption_and_select`（Python 3.12，torch 2.9.1 cu128）

## 安装

```powershell
conda create -n image_caption_and_select python=3.12
conda activate image_caption_and_select
# GPU 版 torch（RTX 50 系需 cu128）：
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# cn_clip 依赖的 lmdb 在 Windows 无预编译包且本项目用不到，需 --no-deps：
python -m pip install --no-deps cn_clip==1.6.0
python -m pip install ftfy regex timm modelscope "PySide6==6.9.2" rapidocr_onnxruntime jieba
```

> ⚠️ **不要安装 PySide6 6.11.1**：该版本的 Qt6Core.dll 缺少 QtCore.pyd 所需的
> 109 个导出符号（QAnyStringView 等），会报
> `ImportError: DLL load failed while importing QtWidgets`（错误 127）。
> 6.9.2 已在本机验证可用。

模型权重（ViT-L-14 约 1.63GB）首次运行自动下载：优先走
[ModelScope](https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-large-patch14)
（国内直连快），失败自动回退 HuggingFace；下载到 `data/models/checkpoints/`。

## 运行

```powershell
conda activate image_caption_and_select
python main.py
```

- 首次启动：选择图片文件夹 → 自动建索引（首次还会下载模型权重）；
  建索引可边建边搜已索引部分，之后"刷新索引"增量更新。
- 命令行工具（无需 GUI）：
  ```powershell
  python tools/index_cli.py index <图片文件夹>   # 建索引（含 OCR）
  python tools/index_cli.py query "海边 日落"    # 文本查询（[文:x] 表示文字命中）
  python tools/index_cli.py stats               # 索引统计
  python tools/smoke_test.py                    # 端到端冒烟测试
  ```

## 项目结构

```
main.py                # 入口：悬浮球 + 查询面板 + 托盘 + 热键 + 后台线程编排
core/
  config.py            # 配置默认值与持久化（data/config.json）
  model.py             # Chinese-CLIP 加载/编码（GPU/CPU 自适应）
  scanner.py           # 文件夹扫描与 mtime 增量对比
  thumbnailer.py       # 缩略图生成（md5(路径) 命名，跨重建稳定）
  store.py             # SQLite 元数据 + npy 向量矩阵（墓碑机制，线程安全）
  searcher.py          # 余弦相似度 top-k
  ocr.py               # RapidOCR 文字识别（PP-OCRv4，CPU）
  textindex.py         # OCR 文本倒排索引（jieba 分词，文字优先匹配）
  pipeline.py          # 与 UI 解耦的核心管线（建索引/查询/OCR 编排）
  worker.py            # QThread 包装（索引线程/搜索线程）
  _win.py              # Windows 引导：注册 conda 环境 DLL 目录（cuDNN 等）
  _qt.py               # Qt 环境引导：QT_PLUGIN_PATH 等
ui/
  floating_ball.py     # 悬浮球（可拖动、右键菜单）
  search_panel.py      # 查询面板（1s 防抖、自适应高度）+ 结果网格 + QDrag 拖拽
  tray.py              # 系统托盘（代码绘制图标）
  hotkey.py            # Windows 全局热键（RegisterHotKey）
tools/
  env_check.py         # 环境探测
  smoke_test.py        # 端到端冒烟测试
  index_cli.py         # 命令行建索引/查询/统计
  *_probe.py / *_diff.py / *_test.py / *_diag.py
                       # DLL/Qt 依赖诊断脚本（本机 Qt 环境排查记录，可复用）
data/                  # 运行时生成（gitignore）：index.db / vectors.npy /
                       # thumbs/ / models/ / config.json
```

## 数据与配置

- `data/index.db`：SQLite 元数据（`ocr` 字段已启用，`caption` 图像描述预留）
- `data/vectors.npy`：图像向量矩阵（10 万张 × 768 维 float32 ≈ 300MB）
- `data/thumbs/`：缩略图缓存（约 15~25KB/张）
- `data/config.json`：图片目录、top-k、模型名、批大小、热键、悬浮球位置等

## 常见问题

- **微信拖拽不生效**：右键结果 → "复制文件"，去微信聊天窗 Ctrl+V 粘贴（等效）。
- **全局热键没反应**：可能被其他软件占用，悬浮球点击始终可用。
- **cn_clip 导入报 flash_attn 错误**：本机 flash-attn 2.8.3 缺少
  `flash_attn.flash_attention` 模块，`core/model.py` 已自动注入桩模块规避
  （仅影响训练路径，推理不受影响）。
- **权重下载慢/失败**：阿里云 OSS 官方链接已 403；程序默认走 ModelScope，
  如仍失败可手动下载 `clip_cn_vit-l-14.pt` 放到 `data/models/checkpoints/`。
- **换大模型（如 ViT-L-14-336）**：改 `data/config.json` 的 `model_name`，
  并删除 `data/index.db` + `data/vectors.npy` 重建索引（维度不同无法复用）。
- **检索慢**：10 万级图片纯 numpy 检索 <50ms；百万级可换 FAISS。

## 开发与调试

- 隐藏调试参数：`python main.py --auto-quit-ms=N`（N 毫秒后自动走正常退出流程）、
  `--auto-search=词A|词B`（启动 3 秒后依次自动触发查询，间隔 1.5 秒），
  用于退出路径与并发压测回归。
- 定位日志：`data/position.log`（查询框每次弹出的坐标计算记录）、
  `data/quit.log`（退出路径分步记录）、`data/skipped.log`（索引/OCR 跳过的文件）。

## 后续可扩展

- 图像描述：用 Qwen2.5-VL 等生成中文描述存入预留的 `images.caption` 字段，
  提升细粒度检索
- 前后端分离：`core/` 已与 UI 解耦，套一层 FastAPI 即可提供 Web 服务
- FAISS：图片量达百万级时替换纯 numpy 检索
