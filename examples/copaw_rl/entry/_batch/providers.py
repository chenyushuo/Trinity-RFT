"""把 CLI / JSON 配置转成 CoPaw 的 provider config。

入口:
    build_provider_config(model_key)            — 给 --models 用
    build_provider_config_from_json(json_item)  — 给 --models-file 用
    sampling_params_to_generate_kwargs(...)     — 共用：JSON 的 sampling_params
                                                  → OpenAI client 可识别的 kwargs
"""

import copy
import json
import os

from .constants import MODELS, PROVIDER_BASE

# OpenAI ChatCompletion 标准字段；其余字段（top_k / min_p / repetition_penalty
# 等 vLLM 私有参数）会被收进 extra_body 透传给后端，OpenAI Python SDK 才不会
# 因为未知顶层 kwargs 而报错。
_OPENAI_STD_SAMPLING_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "stop",
        "seed",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "stream",
        "response_format",
    }
)


def sampling_params_to_generate_kwargs(sampling_params: dict) -> dict:
    """JSON 里的 sampling_params → OpenAI client 可识别的 generate_kwargs。

    标准字段直接放顶层；非标字段（vLLM/dashscope 私有）塞 extra_body。
    auto_eval.py 与 batch_run.py 共用。
    """
    std: dict = {}
    extra: dict = {}
    for k, v in sampling_params.items():
        if k in _OPENAI_STD_SAMPLING_FIELDS:
            std[k] = v
        else:
            extra[k] = v
    if extra:
        std["extra_body"] = extra
    return std


def _ensure_extra_model(cfg: dict) -> None:
    """把 active_llm 指向的模型加进 provider.extra_models，CoPaw 才能找到。"""
    active = cfg.get("active_llm", {})
    pid = active.get("provider_id", "")
    mid = active.get("model", "")
    if not (pid and mid and pid in cfg.get("providers", {})):
        return
    provider = cfg["providers"][pid]
    if any(m.get("id") == mid for m in provider.get("extra_models", [])):
        return
    provider["extra_models"] = [
        *provider.get("extra_models", []),
        {
            "id": mid,
            "name": mid,
            "supports_multimodal": True,
            "supports_image": True,
            "supports_video": False,
        },
    ]


def _build_custom_provider(
    provider_id: str,
    model_id: str,
    name: str,
    base_url: str,
    api_key: str,
) -> dict:
    """auto_eval / JSON 自定义 base_url 时构造的 custom provider 描述。"""
    return {
        "id": provider_id,
        "name": name,
        "default_base_url": "",
        "api_key_prefix": "",
        "models": [
            {
                "id": model_id,
                "name": name,
                "supports_multimodal": True,
                "supports_image": True,
                "supports_video": False,
            }
        ],
        "base_url": base_url,
        "api_key": api_key,
        "chat_model": "OpenAIChatModel",
    }


def build_provider_config(model_key: str) -> dict:
    """根据 model_key 构造 provider config。

    优先级:
        1. auto_eval.py 注入的 AUTO_EVAL_BASE_URL + AUTO_EVAL_MODEL_ID
        2. MODELS 预设表
        3. provider_id:model_name 自定义格式
        4. 兜底走 dashscope
    """
    auto_base_url = os.environ.get("AUTO_EVAL_BASE_URL")
    auto_model_id = os.environ.get("AUTO_EVAL_MODEL_ID")
    if auto_base_url and auto_model_id and model_key == os.environ.get("AUTO_EVAL_MODEL_KEY"):
        cfg = copy.deepcopy(PROVIDER_BASE)
        provider_id = f"auto-{model_key}"
        custom = _build_custom_provider(provider_id, auto_model_id, model_key, auto_base_url, "")
        generate_kwargs_str = os.environ.get("AUTO_EVAL_GENERATE_KWARGS")
        if generate_kwargs_str:
            try:
                custom["generate_kwargs"] = json.loads(generate_kwargs_str)
            except json.JSONDecodeError:
                pass
        cfg["custom_providers"][provider_id] = custom  # type: ignore[index]
        cfg["active_llm"] = {"provider_id": provider_id, "model": auto_model_id}
        return cfg

    if model_key in MODELS:
        active = MODELS[model_key]
    elif ":" in model_key:
        provider_id, model_name = model_key.split(":", 1)
        active = {"provider_id": provider_id, "model": model_name}
    else:
        active = {"provider_id": "dashscope", "model": model_key}

    cfg = copy.deepcopy(PROVIDER_BASE)
    cfg["active_llm"] = active
    _ensure_extra_model(cfg)
    return cfg


def build_provider_config_from_json(item: dict) -> dict:
    """从 --models-file 的一行 JSON 构造 provider config。

    支持两种格式:
        新格式: {"key": "...", "base_url": "...", "model_id": "...", "api_key": ""}
        旧格式: {"key": "...", "provider_id": "dashscope", "model": "qwen3.5-plus"}
    """
    key = item["key"]

    if "base_url" in item:
        cfg = copy.deepcopy(PROVIDER_BASE)
        provider_id = f"json-{key}"
        model_id = item.get("model_id", key)
        cfg["custom_providers"][provider_id] = _build_custom_provider(  # type: ignore[index]
            provider_id, model_id, key, item["base_url"], item.get("api_key", "")
        )
        cfg["active_llm"] = {"provider_id": provider_id, "model": model_id}
        return cfg

    if "provider_id" in item and "model" in item:
        cfg = build_provider_config(key)
        cfg["active_llm"] = {
            "provider_id": item["provider_id"],
            "model": item["model"],
        }
        return cfg

    return build_provider_config(key)
