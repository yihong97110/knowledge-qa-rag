"""LLM 客户端工具

基于 OpenAI SDK（兼容 DashScope），提供：
- get_llm()         返回 OpenAI 客户端单例
- chat()            同步对话，返回完整文本
- chat_stream()     流式对话，逐 token 生成
- chat_json()       要求 LLM 返回 JSON 并解析为 dict
"""
from __future__ import annotations

import json
from typing import Generator, Optional

from openai import OpenAI

from core.config import get_settings


# 全局单例客户端，避免重复创建连接
_llm_client: Optional[OpenAI] = None


def get_llm() -> OpenAI:
    """返回 OpenAI 客户端单例，复用连接。"""
    global _llm_client
    if _llm_client is None:
        settings = get_settings()
        _llm_client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base,
        )
    return _llm_client


def _extra_body() -> dict:
    """构造额外请求体。

    火山方舟 DeepSeek 系列默认开启深度思考（thinking），会显著增加
    首字延迟与 token 消耗；LLM_DISABLE_THINKING=true 时在请求层关闭。
    """
    settings = get_settings()
    if settings.llm_disable_thinking:
        return {"thinking": {"type": "disabled"}}
    return {}


def chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """同步对话，返回完整回复文本。

    Args:
        messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "..."}]
        model: 模型名，默认取配置中的 llm_model
        temperature: 采样温度，默认取配置中的 llm_temperature
        max_tokens: 最大生成 token 数

    Returns:
        LLM 的完整回复文本
    """
    settings = get_settings()
    client = get_llm()
    response = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens,
        extra_body=_extra_body(),
    )
    return response.choices[0].message.content or ""


def chat_stream(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """流式对话，逐 token 生成输出。

    用法::

        for token in chat_stream(messages):
            print(token, end="", flush=True)

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名，默认取配置中的 llm_model
        temperature: 采样温度，默认取配置中的 llm_temperature
        max_tokens: 最大生成 token 数

    Yields:
        每个 token 片段的文本
    """
    settings = get_settings()
    client = get_llm()
    stream = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens,
        stream=True,
        extra_body=_extra_body(),
    )
    # 迭代 choices[0].delta.content，逐 token 产出
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def chat_json(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.0,
    seed: int = 42,
) -> dict:
    """要求 LLM 返回 JSON 并解析为 dict。

    四层保障：
    1. temperature=0 —— 消除采样随机性，同类输入尽量产出同类结果；
    2. seed —— 固定采样种子，尽可能抵消服务端调度/浮点非结合性带来的差异
       （注意：托管 API 均不承诺逐字节确定，OpenAI 官方文档同此口径）；
    3. response_format=json_object —— 服务端约束解码，从协议层保证输出是合法
       JSON（区别于在 prompt 里"请求"模型输出 JSON）；
    4. 系统提示词 JSON 指令 + 容错解析 —— 兜底 response_format 不被支持的场景。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名，默认取配置中的 llm_model
        temperature: 采样温度，结构化任务默认 0
        seed: 采样种子，固定值可提升同输入输出稳定性

    Returns:
        解析后的字典
    """
    settings = get_settings()
    client = get_llm()

    # 在 system prompt 里强调只输出 JSON（不修改入参 messages）
    json_hint = "只输出JSON，不要包含解释性文字或markdown代码块标记。"
    prepared: list[dict] = []
    system_added = False
    for m in messages:
        if m.get("role") == "system":
            prepared.append({**m, "content": f"{m.get('content', '')}\n{json_hint}"})
            system_added = True
        else:
            prepared.append(dict(m))
    if not system_added:
        prepared.insert(0, {"role": "system", "content": json_hint})

    # 先按约束解码调用；若服务端不支持 response_format 则降级重试
    try:
        response = client.chat.completions.create(
            model=model or settings.llm_model,
            messages=prepared,
            temperature=temperature,
            seed=seed,
            response_format={"type": "json_object"},
            extra_body=_extra_body(),
        )
    except Exception:
        response = client.chat.completions.create(
            model=model or settings.llm_model,
            messages=prepared,
            temperature=temperature,
            seed=seed,
            extra_body=_extra_body(),
        )
    content = response.choices[0].message.content or ""

    return _parse_json(content)


def _parse_json(text: str) -> dict:
    """容错解析 JSON。

    1. 先直接 json.loads；
    2. 失败则用 json.JSONDecoder.raw_decode 从首个 { 或 [ 处解析，
       兼容 LLM 在 JSON 前后夹杂多余文字的情况。
    """
    text = text.strip()

    # 去除可能的 markdown 代码块包裹（```json ... ```）
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 1. 直接解析
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. 用 JSONDecoder 从首个 { 或 [ 处容错解析
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    raise ValueError(f"无法解析为 JSON: {text!r}")
