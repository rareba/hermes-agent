"""Deterministic webhook acknowledgements supplied by an owning-profile plugin."""
import asyncio
import inspect
import json
import logging


logger = logging.getLogger(__name__)
CALLBACK_TIMEOUT_SECONDS = 10
MAX_RESPONSE_BYTES = 65536
_INCOMPATIBLE_FIELDS = frozenset({
    "prompt", "script", "cron_job", "deliver", "deliver_only", "coalesce", "skills",
    "toolsets", "deliver_extra", "mirror_to_session",
})


def validate_plugin_hook(name: str, route: dict) -> None:
    if "plugin_hook" not in route:
        return
    hook = route["plugin_hook"]
    if not isinstance(hook, str) or not hook.strip():
        raise ValueError(f"Route '{name}' plugin_hook must be a nonblank string")
    conflicts = _INCOMPATIBLE_FIELDS.intersection(route)
    if conflicts:
        raise ValueError(f"Route '{name}' plugin_hook conflicts with: {', '.join(sorted(conflicts))}")


def _json_value(value, depth=0):
    if depth > 32:
        raise ValueError("Plugin response exceeds JSON depth limit")
    if type(value) is dict:
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Plugin response keys must be strings")
            _json_value(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _json_value(child, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("Plugin response contains non-JSON values")


async def dispatch_plugin_hook(hook: str, *, route_name: str, profile: str | None, payload,
                               delivery_id: str):
    """No observer-style error swallowing: only the single consumer can acknowledge receipt."""
    from aiohttp import web
    from hermes_cli.plugins import _delivery_manager
    from hermes_cli.profiles import get_active_profile_name

    try:
        profile = profile or get_active_profile_name()
        callbacks = tuple(_delivery_manager().iter_hook_callbacks(hook))
        if len(callbacks) != 1 or not inspect.iscoroutinefunction(callbacks[0]):
            raise ValueError("Webhook plugin hook requires exactly one async callback")
        result = await asyncio.wait_for(callbacks[0](
            route_name=route_name, profile=profile, payload=payload, delivery_id=delivery_id),
            timeout=CALLBACK_TIMEOUT_SECONDS)
        if not isinstance(result, dict) or type(result.get("status")) is not int or not 200 <= result["status"] <= 599:
            raise ValueError("Plugin response requires an HTTP status from 200 through 599")
        if type(result.get("body")) is not dict:
            raise ValueError("Plugin response body must be a JSON object")
        _json_value(result["body"])
        encoded = json.dumps(result["body"], allow_nan=False, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_RESPONSE_BYTES:
            raise ValueError("Plugin response body exceeds 64 KiB")
        return web.Response(body=encoded, content_type="application/json", status=result["status"])
    except Exception:
        logger.exception("[webhook] Plugin hook failed route=%s hook=%s", route_name, hook)
        return web.json_response({"error": "Webhook plugin unavailable"}, status=503)
