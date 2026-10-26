"""MAAS client (OpenAI-compatible LiteLLM proxy on genailab.tcs.in).

Only trip details and public dataset text are sent to the model.
If MAAS is not configured or a call fails, chat() returns None and callers
use their offline text instead.

Connectivity check:  python -m src.llm
"""
import re

from src import config

_client = None
last_error = ""


def is_configured():
    return bool(config.MAAS_BASE_URL and config.MAAS_API_KEY and config.MAAS_MODEL)


def _get_client():
    global _client
    if _client is None:
        import httpx
        from openai import OpenAI
        http = httpx.Client(verify=config.MAAS_VERIFY_SSL, timeout=config.MAAS_TIMEOUT)
        _client = OpenAI(base_url=config.MAAS_BASE_URL, api_key=config.MAAS_API_KEY,
                         http_client=http, max_retries=1)
    return _client


def _is_reasoning_model(name):
    name = name.lower()
    return "gpt-5" in name or bool(re.search(r"(^|[/-])o[134]", name))


def _clean(text):
    # DeepSeek-R1 style models put their reasoning inside <think> tags
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def describe_error(e):
    """Error text including the root cause (openai wraps SSL / network errors)."""
    msg = f"{type(e).__name__}: {e}"
    root = e
    while root.__cause__ is not None:
        root = root.__cause__
    if root is not e:
        msg += f" ({type(root).__name__}: {root})"
    return msg


def chat(system, user, temperature=0.3, max_tokens=800, history=None):
    """history: earlier turns as [(role, text), ...] with role 'user' or 'assistant'."""
    global last_error
    if not is_configured():
        last_error = "MAAS is not configured (.env)"
        return None

    messages = [{"role": "system", "content": system}]
    messages += [{"role": r, "content": t} for r, t in (history or [])]
    messages.append({"role": "user", "content": user})
    if _is_reasoning_model(config.MAAS_MODEL):
        # gpt-5 family: no custom temperature, and the token limit includes reasoning tokens.
        # sent as extra_body so older openai package versions accept them too
        params = {"extra_body": {"max_completion_tokens": max_tokens * 4, "reasoning_effort": "low"}}
    else:
        params = {"temperature": temperature, "max_tokens": max_tokens}

    try:
        client = _get_client()
        for attempt in (params, {}):
            try:
                resp = client.chat.completions.create(model=config.MAAS_MODEL, messages=messages, **attempt)
            except Exception as e:
                if attempt and getattr(e, "status_code", None) in (400, 422):
                    continue  # model rejected a parameter, retry with defaults
                raise
            text = _clean(resp.choices[0].message.content)
            if text:
                last_error = ""
                return text
        last_error = "model returned an empty answer"
    except Exception as e:
        last_error = describe_error(e)
    print(f"[llm] MAAS call failed: {last_error}")
    return None


def list_models():
    return sorted(m.id for m in _get_client().models.list().data)


def check():
    print(f"Endpoint : {config.MAAS_BASE_URL}")
    print(f"Model    : {config.MAAS_MODEL}")
    if not is_configured():
        print("MAAS_API_KEY / MAAS_MODEL missing - fill in .env first.")
        return False
    try:
        models = list_models()
        print(f"Models available to this key ({len(models)}):")
        for m in models:
            print("   ", m)
        if config.MAAS_MODEL not in models:
            print(f"WARNING: '{config.MAAS_MODEL}' is not in the list above.")
    except Exception as e:
        print(f"Could not list models: {describe_error(e)}")
    reply = chat("You are a helpful assistant.", "Hi")
    if reply:
        print(f"Reply    : {reply}")
        print("MAAS connection OK")
        return True
    print(f"FAILED   : {last_error}")
    return False


if __name__ == "__main__":
    check()
