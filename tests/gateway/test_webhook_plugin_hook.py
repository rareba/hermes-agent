"""Deterministic plugin ingress returns its durable consumer's acknowledgement."""
import asyncio
import base64
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


_SECRET = "plugin-test-secret"


def _headers(body, delivery="delivery-1"):
    stamp = str(int(time.time()))
    signature = base64.b64encode(hmac.new(
        _SECRET.encode(), f"{delivery}.{stamp}.".encode() + body, hashlib.sha256).digest()).decode()
    return {"webhook-id": delivery, "webhook-timestamp": stamp,
            "webhook-signature": "v1," + signature, "Content-Type": "application/json"}


def _adapter(route=None):
    return WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0,
        "routes": {"receipt": {"secret": _SECRET, "plugin_hook": "receipt_ingress", **(route or {})}}}))


def _app(adapter):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    app.router.add_post("/p/{profile}/webhooks/{route_name}", adapter._handle_webhook)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "coder"])
async def test_authenticated_plugin_ack_owns_retry_and_profile(monkeypatch, profile):
    from hermes_cli import plugins
    from hermes_constants import get_hermes_home
    from hermes_cli.profiles import get_profile_dir

    adapter = _adapter({"profile": profile, "events": ["receipt"],
                        "filters": {"field": "value", "equals": 7}})
    adapter.gateway_runner = MagicMock(config=MagicMock(multiplex_profiles=True))
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve",
                        lambda **kwargs: [(name, get_profile_dir(name)) for name in ("default", "coder", "other")])
    url = "/webhooks/receipt" if profile == "default" else "/p/coder/webhooks/receipt"
    adapter.handle_message = AsyncMock(side_effect=AssertionError("agent must not run"))
    adapter._render_prompt = MagicMock(side_effect=AssertionError("prompt must not render"))
    callback = AsyncMock(return_value={"status": 201, "body": {"receipt": "persisted"}})
    managers = []

    def manager():
        managers.append(get_hermes_home())
        return MagicMock(iter_hook_callbacks=MagicMock(return_value=(callback,)))

    monkeypatch.setattr(plugins, "_delivery_manager", manager)
    body = json.dumps({"type": "receipt", "value": 7}).encode()
    async with TestClient(TestServer(_app(adapter))) as client:
        for scheme in ("webhook", "svix"):
            signed_headers = {key.replace("webhook-", f"{scheme}-"): value for key, value in _headers(body).items()}
            signed_headers["X-GitHub-Delivery"] = "unsigned-conflicting-id"
            if scheme == "svix":
                signed_headers["webhook-id"] = "unsigned-conflicting-standard-id"
            response = await client.post(url, data=body, headers=signed_headers)
            assert response.status == 201
            assert await response.json() == {"receipt": "persisted"}
        for ignored in ({"type": "other", "value": 7}, {"type": "receipt", "value": 8}):
            ignored_body = json.dumps(ignored).encode()
            response = await client.post(url, data=ignored_body, headers=_headers(ignored_body))
            assert response.status == 200
            assert (await response.json())["status"] == "ignored"
        bad_headers = _headers(body)
        bad_headers["webhook-signature"] = "v1,invalid"
        assert (await client.post(url, data=body, headers=bad_headers)).status == 401
        assert (await client.post("/p/other/webhooks/receipt", data=body, headers=_headers(body))).status == 404
        # A valid body-only signature authenticates, but cannot substitute for a stable ID.
        headers = {"X-Hub-Signature-256": "sha256=" + hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()}
        assert (await client.post(url, data=body, headers=headers)).status == 400
    assert callback.await_count == 2
    callback.assert_awaited_with(route_name="receipt", profile=profile,
                                 payload={"type": "receipt", "value": 7}, delivery_id="delivery-1")
    assert len(managers) == 2
    assert managers[0] == managers[1] == get_profile_dir(profile)
    assert not adapter._seen_deliveries
    adapter.handle_message.assert_not_called()
    adapter._render_prompt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "multiple", "sync", "exception", "timeout", "invalid",
                                     "nan", "oversize", "status", "keys"])
async def test_plugin_failures_are_retryable_without_agent(monkeypatch, failure):
    from hermes_cli import plugins
    from gateway.platforms import webhook_plugin_hook as hook

    adapter = _adapter()
    adapter.handle_message = AsyncMock(side_effect=AssertionError("agent must not run"))

    async def callback(**kwargs):
        if failure == "exception":
            raise RuntimeError("consumer unavailable")
        if failure == "timeout":
            await asyncio.Event().wait()
        body = {"nan": float("nan")} if failure == "nan" else {"ok": True}
        if failure == "oversize":
            body = {"huge": "x" * 65537}
        if failure == "keys":
            body = {1: "non-string"}
        return None if failure == "invalid" else {"status": True if failure == "status" else 200, "body": body}

    handlers = {"missing": (), "multiple": (callback, callback), "sync": (lambda **kwargs: {},)}.get(
        failure, (callback,))
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: MagicMock(
        iter_hook_callbacks=MagicMock(return_value=handlers)))
    monkeypatch.setattr(hook, "CALLBACK_TIMEOUT_SECONDS", 0.01)
    body = b'{"type":"receipt"}'
    async with TestClient(TestServer(_app(adapter))) as client:
        for _ in range(2):
            response = await client.post("/webhooks/receipt", data=body, headers=_headers(body))
            assert response.status == 503
    assert not adapter._seen_deliveries
    adapter.handle_message.assert_not_called()


@pytest.mark.parametrize("conflict", ["prompt", "script", "cron_job", "deliver", "deliver_only", "coalesce",
                                      "skills", "toolsets", "deliver_extra", "mirror_to_session"])
def test_plugin_modes_validated_for_startup_and_dynamic_routes(conflict):
    adapter = _adapter({conflict: False})
    route = adapter._routes["receipt"]
    with pytest.raises(ValueError, match="plugin_hook"):
        adapter._validate_route("receipt", route)
    assert not adapter._dynamic_route_allowed("receipt", route)


@pytest.mark.parametrize("name", [None, "", "  ", 3])
def test_plugin_hook_requires_a_name(name):
    adapter = _adapter({"plugin_hook": name})
    with pytest.raises(ValueError, match="plugin_hook"):
        adapter._validate_route("receipt", adapter._routes["receipt"])
    assert not adapter._dynamic_route_allowed("receipt", adapter._routes["receipt"])
