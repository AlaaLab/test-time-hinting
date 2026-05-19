import asyncio
import base64
import json
import mimetypes
import re


def _img(p):
    mime = mimetypes.guess_type(p)[0] or "image/png"
    with open(p, "rb") as f:
        return f.read(), mime


class OpenAIClient:
    def __init__(self):
        from openai import AsyncOpenAI
        self.c = AsyncOpenAI()

    async def call(self, *, image_path, system_prompt, user_prompt, model, **kw):
        data, mime = _img(image_path)
        b64 = base64.b64encode(data).decode("utf-8")
        url = f"data:{mime};base64,{b64}"
        content = [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": url},
        ]
        req = {
            "model": model,
            "instructions": system_prompt,
            "input": [{"role": "user", "content": content}],
        }
        re_eff = kw.pop("reasoning_effort", None)
        re_sum = kw.pop("reasoning_summary", None)
        if re_eff is not None or re_sum is not None:
            r = {}
            if re_eff is not None:
                r["effort"] = re_eff
            if re_sum is not None:
                r["summary"] = re_sum
            req["reasoning"] = r
        for k in ("thinking", "thinking_budget", "thinking_budget_tokens"):
            kw.pop(k, None)
        timeout = kw.pop("timeout", None)
        kw.pop("max_retries", None)
        req.update(kw)
        coro = self.c.responses.create(**req)
        if timeout is not None:
            resp = await asyncio.wait_for(coro, timeout=float(timeout))
        else:
            resp = await coro
        return getattr(resp, "output_text", "") or ""


class AnthropicClient:
    def __init__(self):
        from anthropic import AsyncAnthropic
        self.c = AsyncAnthropic()

    async def call(self, *, image_path, system_prompt, user_prompt, model, **kw):
        data, mime = _img(image_path)
        b64 = base64.b64encode(data).decode("utf-8")
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
        ]
        req = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }
        if "max_output_tokens" in kw:
            req["max_tokens"] = kw.pop("max_output_tokens")
        thinking_mode = kw.pop("thinking", None)
        tb = kw.pop("thinking_budget_tokens", None)
        if tb is None:
            tb = kw.pop("thinking_budget", None)
        else:
            kw.pop("thinking_budget", None)
        if thinking_mode == "enabled":
            d = {"type": "enabled"}
            if tb is not None:
                d["budget_tokens"] = int(tb)
            req["thinking"] = d
        elif thinking_mode in ("disabled", "off", "none"):
            req["thinking"] = {"type": "disabled"}
        for k in ("reasoning_effort", "reasoning_summary"):
            kw.pop(k, None)
        timeout = kw.pop("timeout", None)
        kw.pop("max_retries", None)
        req.update(kw)
        coro = self.c.messages.create(**req)
        if timeout is not None:
            resp = await asyncio.wait_for(coro, timeout=float(timeout))
        else:
            resp = await coro
        out = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                out.append(getattr(block, "text", "") or "")
        return "\n".join(out).strip()


class GeminiClient:
    def __init__(self):
        from google import genai
        from google.genai import types
        self.types = types
        self.c = genai.Client()

    async def call(self, *, image_path, system_prompt, user_prompt, model, **kw):
        data, mime = _img(image_path)
        parts = [user_prompt, self.types.Part.from_bytes(data=data, mime_type=mime)]
        cfg_kw = {"system_instruction": system_prompt}
        tb = kw.pop("thinking_budget", None)
        thinking = kw.pop("thinking", None)
        if tb is None and isinstance(thinking, str) and thinking.lower() == "dynamic":
            tb = -1
        kw.pop("thinking_budget_tokens", None)
        if tb is not None:
            cfg_kw["thinking_config"] = self.types.ThinkingConfig(thinking_budget=int(tb))
        for k in ("reasoning_effort", "reasoning_summary"):
            kw.pop(k, None)
        timeout = kw.pop("timeout", None)
        kw.pop("max_retries", None)
        gen_cfg = self.types.GenerateContentConfig(**cfg_kw, **kw)
        coro = self.c.aio.models.generate_content(model=model, contents=parts, config=gen_cfg)
        if timeout is not None:
            resp = await asyncio.wait_for(coro, timeout=float(timeout))
        else:
            resp = await coro
        return (getattr(resp, "text", None) or "").strip()


def get_client(provider):
    p = (provider or "").strip().lower()
    if p == "openai":
        return OpenAIClient()
    if p == "anthropic":
        return AnthropicClient()
    if p == "gemini":
        return GeminiClient()
    raise ValueError(f"unknown provider: {provider!r}")


def extract_json(text):
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None
