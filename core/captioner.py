"""图像描述（caption）管线：结构化 JSON schema 唯一定义 + 本地/云端实现。

设计要点（借鉴 image-features 的 ImageCaption，规避其坑）：
- schema 收敛为 7 个顶层字段（去掉信息重叠的 dimensions），生成端与
  消费端共用本文件定义，杜绝"提示词漂移"。
- 字段一律 .get() 兜底，caption JSON 解析失败返回 {}，绝不抛 KeyError。
- 本地实现优先（Qwen3-VL-2B 默认 / Qwen2.5-VL-7B 可选）；API 实现
  仅手动触发（config api_manual_only）。
- 单图失败向上抛 CaptionerError，由 pipeline 逐图隔离。
"""
import base64
import json
import re
import textwrap

CAPTION_FIELDS = (
    "background",    # 背景环境描述
    "colors",        # 主色调及分布
    "style",         # 艺术风格/画风
    "content",       # 整体内容概述
    "people",        # [{description, action, special_name}]
    "objects",       # [{name, description, location}]
    "text_elements", # 图中所有可见文字
)

CAPTION_PROMPT = textwrap.dedent("""
    请严格按以下规范输出：
    【输出要求】
    1. 仅输出标准JSON，无任何前缀/后缀/注释/代码块标记（如```json```）
    2. JSON必须包含且仅包含以下字段：
    {
    "background": "背景环境描述（如'黄昏的都市天台'），无则''",
    "colors": "主色调及分布（如'主色：靛蓝裙摆+暖黄灯光，天空渐变紫红'）",
    "style": "艺术风格/画风（如'赛博朋克插画'、'水墨国风'、'3D卡通渲染'、'胶片摄影'），非画质描述",
    "content": "整体内容简洁概述（避免与细节重复）",
    "people": [
        {
        "description": "外貌特征（性别/年龄/衣着等）",
        "action": "【必须详细】动作+姿态+表情+朝向（例：'右手扶帽檐仰头大笑，左脚踏在木箱上'）",
        "special_name": "图片中可见的人物标识名称（如服装印'孙悟空'、标签写'李清照'），无则''"
        }
    ],
    "objects": [
        {
        "name": "【优先使用图片专有名称】如瓶身'农夫山泉'、招牌'知味观'；无则通用描述",
        "description": "外观细节（材质/形状/状态）",
        "location": "图中位置（如'左下角茶几'、'人物手持'）"
        }
    ],
    "text_elements": "所有可见文字内容（如海报标语'春日特惠'、服装印花'勇气'），无则''"
    }
    【铁律】
    仅描述视觉可见内容！禁止：
    - 推测（"可能""似乎"）、评价（"美观""专业"）、技术分析（"构图优秀""拍摄于黄昏"）
    - 添加"根据图片""描述如下"等引导语
    名称替换原则：人物/物品名称必须严格依据图片中可见文字标识填写
    空值规范：字符串字段无内容填""，数组无内容填[]（如无人物： "people": []）
    【执行】
    请直接输出纯净JSON。
    """).strip()


class CaptionerError(Exception):
    """单张图片描述失败（由 pipeline 逐图隔离记录）。"""


def clean_caption_json(raw: str) -> dict:
    """清洗大模型返回内容并解析为 dict；失败返回 {}。"""
    if not raw or not isinstance(raw, str):
        return {}
    s = raw.strip()
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```", "", s)
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        s = m.group(0)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_caption(data: dict) -> dict:
    """把模型输出规整为 schema 形态（全部字段存在、字符串化）。"""
    out = {}
    for f in ("background", "colors", "style", "content", "text_elements"):
        v = data.get(f, "")
        out[f] = str(v).strip() if isinstance(v, str) else ""
    people = []
    for p in (data.get("people") or []):
        if not isinstance(p, dict):
            continue
        people.append({
            "description": str(p.get("description", "") or "").strip(),
            "action": str(p.get("action", "") or "").strip(),
            "special_name": str(p.get("special_name", "") or "").strip(),
        })
    objects = []
    for o in (data.get("objects") or []):
        if not isinstance(o, dict):
            continue
        objects.append({
            "name": str(o.get("name", "") or "").strip(),
            "description": str(o.get("description", "") or "").strip(),
            "location": str(o.get("location", "") or "").strip(),
        })
    out["people"] = people
    out["objects"] = objects
    return out


class LocalVlmCaptioner:
    """本地 VLM 图像描述（transformers，Qwen3-VL / Qwen2.5-VL 系列）。"""

    def __init__(self, model_name: str, device: str = "auto",
                 model_dir: str = "", max_new_tokens: int = 1024):
        self._model_name = model_name
        self._device_pref = device
        self._model_dir = model_dir
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._dev = None

    def _load(self):
        # 必须先于 import torch：注册 conda 环境的 DLL 目录（cuDNN 等），
        # 未激活环境直接运行 python 时 torch 找不到 cudnn DLL 会崩溃
        from core._win import ensure_env_dlls
        ensure_env_dlls()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from core.hub import resolve_pretrained
        dev = "cuda" if (self._device_pref in ("cuda", "auto")
                         and torch.cuda.is_available()) else "cpu"
        name = resolve_pretrained(self._model_name, self._model_dir)
        dtype = torch.float16 if dev == "cuda" else torch.float32
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                name, dtype=dtype, trust_remote_code=True).eval().to(dev)
        except TypeError:
            # transformers < 4.56 用 torch_dtype
            model = AutoModelForImageTextToText.from_pretrained(
                name, torch_dtype=dtype, trust_remote_code=True).eval().to(dev)
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self._model, self._processor, self._dev = model, processor, dev

    def _ensure(self):
        if self._model is None:
            self._load()

    def unload(self):
        self._model = None
        self._processor = None
        self._dev = None

    def _generate(self, image, max_new_tokens: int) -> str:
        import torch
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text], images=[image],
            return_tensors="pt").to(self._dev)
        with torch.no_grad():
            gen = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = gen[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            gen, skip_special_tokens=True)[0]

    def describe(self, image) -> dict:
        """PIL 图像 -> 规整后的 caption dict。失败抛 CaptionerError。

        若因生成上限截断导致 JSON 解析失败，自动以 1.5 倍上限重试一次。
        """
        self._ensure()
        try:
            raw = self._generate(image, self._max_new_tokens)
        except Exception as e:
            raise CaptionerError(f"VLM 推理失败: {e}") from e
        data = clean_caption_json(raw)
        if not data and self._max_new_tokens < 1536:
            # 疑似 JSON 被截断：加大上限重试一次
            try:
                raw = self._generate(image, int(self._max_new_tokens * 1.5))
            except Exception as e:
                raise CaptionerError(f"VLM 重试失败: {e}") from e
            data = clean_caption_json(raw)
        if not data:
            raise CaptionerError(f"caption 输出无法解析为 JSON: {raw[:120]!r}")
        return normalize_caption(data)


class ApiCaptioner:
    """DashScope qwen3-vl-plus 云端描述（仅手动触发，见 config）。"""

    def __init__(self, base_url: str =
                 "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self._base_url = base_url

    def describe(self, image) -> dict:
        import io
        try:
            from openai import OpenAI
        except ImportError as e:
            raise CaptionerError("缺少 openai 依赖，无法调用云端 API") from e
        import os
        api_key = os.getenv("ALIYUN_API_KEY")
        if not api_key or "xxxx" in api_key:
            raise CaptionerError("未配置有效的 ALIYUN_API_KEY")
        buf = io.BytesIO()
        img = image.convert("RGB")
        img.thumbnail((1024, 1024))
        img.save(buf, "JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        client = OpenAI(api_key=api_key, base_url=self._base_url)
        try:
            resp = client.chat.completions.create(
                model="qwen3-vl-plus",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "你是一名严谨的图像内容解析器"},
                    {"role": "user", "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]},
                ],
            )
        except Exception as e:
            raise CaptionerError(f"云端 API 调用失败: {e}") from e
        data = clean_caption_json(resp.choices[0].message.content)
        if not data:
            raise CaptionerError("云端返回无法解析为 JSON")
        return normalize_caption(data)


def make_captioner(model_name: str, device: str = "auto",
                   model_dir: str = "", max_new_tokens: int = 1024) -> LocalVlmCaptioner:
    """创建本地 captioner，并注册到 model_manager（批次结束可统一卸载）。"""
    from core import model_manager
    mgr = model_manager.ModelManager.get()
    slot = mgr.slot("vlm", idle_seconds=0)
    if not slot.is_loaded():
        cap = LocalVlmCaptioner(model_name, device, model_dir,
                                max_new_tokens=max_new_tokens)
        slot.configure(lambda: cap, unload_fn=cap.unload)
    return slot.acquire()


def make_api_captioner() -> ApiCaptioner:
    return ApiCaptioner()
