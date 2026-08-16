# 图片语义搜索工具（本地 Chinese-CLIP + PySide6 悬浮球）

一个完全本地运行的图片语义搜索工具：屏幕上方常驻悬浮球，点击（或
`Ctrl+Shift+空格`）弹出查询框，输入中文描述（停止输入 1 秒自动搜索），
显示最匹配的图片缩略图，可直接**拖拽到微信聊天**发送。

## 特性

- **完全本地**：Chinese-CLIP ViT-L/14 语义检索 + PP-OCRv4 文字识别，无任何云端 API
- **中文自然语言搜图**：支持空格分隔多关键词（如"海边 日落"，取词向量平均）
- **图片文字优先匹配**：后台 OCR 识别图中文字建立倒排索引，查询词命中图中
  文字的结果排最前（角标"文"），其余按语义相似度补足；悬停可见命中的文字
- **本地图像描述增强（第二期）**：后台用本地 VLM（默认 Qwen3-VL-2B，8GB 卡
  档位）为图片生成结构化中文描述（背景/颜色/画风/内容/人物/物体/图中文字），
  描述经 Qwen3-Embedding-0.6B 编码为字段化行向量，与 CLIP 两路做 RRF 融合
  检索，细粒度查询（"穿红衣服的人"、"左下角的杯子"）显著增强；
  caption 向量命中的结果带绿色角标"述"，悬停显示命中的描述行
- **模型空闲卸载**：CLIP/嵌入模型默认常驻（查询速度优先），`config.json` 的
  `idle_unload_seconds` 设为 >0 可启用空闲自动卸载；VLM 在描述批次结束后必卸
- **查询取消**：输入变化即取消在途旧查询（版本号丢弃过期结果），不浪费 GPU
- **悬浮球**：置顶、可拖动、记住位置；右键菜单：刷新索引 / 显示数量（5/10/20/50）/ 退出
- **查询面板跟随悬浮球**：水平居中于球下方（空间不足自动翻到上方，拖动球时
  实时跟随）；**无搜索时收缩为仅输入框**，出结果才展开图片区
- **系统托盘**：右键可显示/隐藏悬浮球、打开查询框、刷新索引、设置显示数量、
  查看索引数量、退出；单击/双击托盘图标呼出查询框
- **结果交互**：单击缩略图打开原图；**按住拖到微信直接发送文件**；右键复制
  文件 / 打开所在文件夹 / 从索引移除；滚轮在面板任意位置均可滚动结果
- **增量索引**："刷新索引"按 mtime 只处理新增/改动图片，中途退出自动续传；
  改动过的图片会全量刷新元数据并重建缩略图；向量文件原子写入（中断不损坏）；
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
- caption 第二期（可选）：`transformers>=4.51`；8GB 卡推荐 Qwen3-VL-2B
  （约 5GB 显存），≥16GB 卡可换 Qwen2.5-VL-7B（fp16 ~16GB / INT4 ~8GB）
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
# caption 第二期（可选，不影响基础检索）：
python -m pip install "transformers>=4.51"
# 若要用 Qwen2.5-VL-7B 的 INT4 量化权重（8GB 卡选装）：python -m pip install auto-gptq
```

> ⚠️ **不要安装 PySide6 6.11.1**：该版本的 Qt6Core.dll 缺少 QtCore.pyd 所需的
> 109 个导出符号（QAnyStringView 等），会报
> `ImportError: DLL load failed while importing QtWidgets`（错误 127）。
> 6.9.2 已在本机验证可用。

模型权重（ViT-L-14 约 1.63GB）首次运行自动下载：优先走
[ModelScope](https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-large-patch14)
（国内直连快），失败自动回退 HuggingFace；下载到 `data/models/checkpoints/`。
caption/嵌入模型的权重同样优先 ModelScope（`core/hub.py`），失败回退 HF。

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
  python tools/index_cli.py query "海边 日落"    # 文本查询（[文:x] 文字命中，[述:x] 描述命中）
  python tools/index_cli.py caption             # 为图片生成结构化描述（本地 VLM，断点续传）
  python tools/index_cli.py embed               # 把描述构建为检索向量（检索第二路）
  python tools/index_cli.py stats               # 索引统计（含 caption 状态）
  python tools/smoke_test.py                    # 端到端冒烟测试
  ```
- caption 第二期启用流程（先 A/B 再全量，见 `tools/vlm_bench.py`）：
  ```powershell
  # 1) 选型评测：30 张样例对比 2B 与 7B 的质量/速度/显存（结果存 data/vlm_bench/）
  python tools/vlm_bench.py --images <样例图片目录> --limit 30 \
      --models Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen2.5-VL-7B-Instruct
  # 2) 按评测结果确认 config.json 的 caption_model（默认 Qwen/Qwen3-VL-2B-Instruct）
  # 3) 全量生成描述（2300 张在 8GB 卡上约 20~60 分钟，可随时 Ctrl+C 后重跑续传）
  python tools/index_cli.py caption
  # 4) 构建描述向量（检索第二路生效）
  python tools/index_cli.py embed
  ```
- caption 融合可一键回退：把 `data/config.json` 的 `caption_search_enabled` 设为
  `false` 即恢复纯 CLIP 检索；`fusion.w_clip / w_caption / rrf_k` 可调融合权重。
- 云端兜底（仅手动触发）：设置环境变量 `ALIYUN_API_KEY` 后，在
  `core/captioner.py` 的 `ApiCaptioner` 基础上手动调用（本工具默认纯本地）。

## 项目结构

```
main.py                # 入口：悬浮球 + 查询面板 + 托盘 + 热键 + 后台线程编排
core/
  config.py            # 配置默认值与持久化（data/config.json，含 caption/融合参数）
  model.py             # Chinese-CLIP 加载/编码（GPU/CPU 自适应，支持卸载）
  model_manager.py     # 模型生命周期：懒加载 / 空闲卸载 / 显存回收（clip/embed/vlm）
  hub.py               # 权重获取：本地优先，ModelScope 优先，回退 HuggingFace
  captioner.py         # 图像描述：schema 唯一定义 + 本地 VLM + 云端(仅手动)
  embed.py             # Qwen3-Embedding 封装（查询指令前缀 + MRL 截断 + 归一化）
  captionindex.py      # caption 字段化行向量索引（两段式：聚合粗排 + 行级精排）
  scanner.py           # 文件夹扫描与 mtime 增量对比
  thumbnailer.py       # 缩略图生成（md5(路径) 命名，force 强制重建）
  store.py             # SQLite 元数据 + npy 向量矩阵（墓碑机制，原子写入，线程安全）
  searcher.py          # 余弦相似度 top-k（含 SearchCancelled）
  ocr.py               # RapidOCR 文字识别（PP-OCRv4，CPU）
  textindex.py         # OCR 文本倒排索引（jieba 分词，文字优先匹配，可取消）
  pipeline.py          # 与 UI 解耦的核心管线（索引/caption/嵌入/三路融合查询）
  worker.py            # QThread 包装（索引/搜索/caption/嵌入线程，查询可取消）
  _win.py              # Windows 引导：注册 conda 环境 DLL 目录（cuDNN 等）
  _qt.py               # Qt 环境引导：QT_PLUGIN_PATH 等
ui/
  floating_ball.py     # 悬浮球（可拖动、右键菜单）
  search_panel.py      # 查询面板（1s 防抖、版本号丢弃过期结果）+ 结果网格 + 拖拽
  tray.py              # 系统托盘（代码绘制图标）
  hotkey.py            # Windows 全局热键（RegisterHotKey）
tools/
  env_check.py         # 环境探测
  smoke_test.py        # 端到端冒烟测试（含 caption 工具函数/原子保存验证）
  index_cli.py         # 命令行：index / query / caption / embed / stats
  vlm_bench.py         # VLM caption A/B 评测（质量样例/速度/显存）
  *_probe.py / *_diff.py / *_test.py / *_diag.py
                       # DLL/Qt 依赖诊断脚本（本机 Qt 环境排查记录，可复用）
data/                  # 运行时生成（gitignore）：index.db / vectors.npy /
                       # caption_vecs.npy / caption_agg.npy / thumbs/ / models/ / config.json
```

## 数据与配置

- `data/index.db`：SQLite 元数据（`ocr` 已启用，`caption` 存结构化描述 JSON，
  `status/error` 记录 caption 失败隔离，`meta` 表记录模型版本/维度）
- `data/vectors.npy`：图像向量矩阵（10 万张 × 768 维 float32 ≈ 300MB）
- `data/caption_vecs.npy`：描述行向量矩阵（每图 ≤12 行 × embed_dim，fp16）
- `data/caption_agg.npy`：每图 max-pool 聚合向量（检索快路径）
- `data/thumbs/`：缩略图缓存（约 15~25KB/张）
- `data/config.json`：图片目录、top-k、模型名、批大小、热键、悬浮球位置、
  caption 模型/维度、融合权重（`fusion.*`）、空闲卸载秒数（`idle_unload_seconds`）等

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
- **caption 显存不足（8GB 卡）**：caption 期间默认卸载嵌入模型腾显存；若仍 OOM，
  把 `config.json` 的 `caption_keep_clip` 设为 `false`（caption 期间搜索会先重载
  CLIP，约 2~5 秒）；VLM 换 2B 或减少后台负载。
- **caption 换模型**：改 `caption_model` 后对 `caption IS NULL` 的行续跑即可；
  已生成描述无需重跑（schema 一致）。若用 `--retry-failed` 只重试失败行。
- **换嵌入维度（embed_dim）**：改配置后重新执行 `python tools/index_cli.py embed`。

## 后续可扩展

- 图搜图：把 `pipeline.search_by_image()` 接入（`searcher.search` 已支持任意
  查询向量），UI 支持拖图片到悬浮球触发
- 前后端分离：`core/` 已与 UI 解耦，套一层 FastAPI 即可提供 Web 服务
- FAISS：图片量达百万级时替换纯 numpy 检索
- 多目录 gallery 隔离与 watchdog 自动增量：借鉴参考项目 backend 的目录指纹设计

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
