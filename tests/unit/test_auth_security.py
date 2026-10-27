"""
tests/unit/test_auth_security.py
---------------------------------
Regression tests for the authentication security layer.

Covers:
  1. No hardcoded secret fallback in source code (B105 regression guard).
  2. AUTH_SECRET_KEY env-var is required — startup fails safely without it.
  3. JWT token creation and decoding round-trips correctly.
  4. Registration, login, and /auth/me endpoints via FastAPI TestClient.
  5. Protected endpoint rejects missing / invalid tokens (401 enforcement).
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# 1. No hardcoded secret fallback — B105 regression guard
# ---------------------------------------------------------------------------

class TestNoHardcodedSecretFallback:
    """Ensure the insecure fallback string was permanently removed from source."""

    def test_auth_py_has_no_hardcoded_fallback(self):
        auth_src = Path(__file__).resolve().parents[2] / "api" / "auth.py"
        content = auth_src.read_text(encoding="utf-8")
        # The specific insecure fallback string must not appear in source
        assert "insecure-dev-fallback" not in content, (
            "Hardcoded insecure fallback found in api/auth.py — B105 regression."
        )

    def test_auth_py_raises_without_secret_key(self, monkeypatch):
        """Importing api.auth with AUTH_SECRET_KEY unset must raise RuntimeError."""
        # Remove the env-var
        monkeypatch.delenv("AUTH_SECRET_KEY", raising=False)
        # Also ensure .env cannot supply it by patching load_dotenv to no-op
        with mock.patch("dotenv.load_dotenv", return_value=None):
            # Force re-import by removing cached module
            for mod_name in list(sys.modules.keys()):
                if "api.auth" in mod_name or mod_name == "api.auth":
                    del sys.modules[mod_name]
            with pytest.raises(RuntimeError, match="AUTH_SECRET_KEY"):
                import api.auth  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Authentication endpoints via TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def auth_client():
    """TestClient with AUTH_SECRET_KEY set to a safe test value."""
    test_secret = "test-secret-for-auth-unit-tests-minimum-length-ok"
    with mock.patch.dict(os.environ, {"AUTH_SECRET_KEY": test_secret}):
        # Reload api.auth to pick up the env-var
        for mod_name in list(sys.modules.keys()):
            if "api.auth" in mod_name or mod_name == "api.auth":
                del sys.modules[mod_name]
        from api.main import app
        with TestClient(app) as client:
            yield client


class TestAuthEndpoints:
    """End-to-end auth workflow through the FastAPI TestClient."""

    _registered_email = "regtest_user@example.com"
    _registered_password = "Secure@Pass1234"
    _token: str = ""

    def test_register_new_user(self, auth_client):
        resp = auth_client.post(
            "/auth/register",
            json={
                "email": self._registered_email,
                "password": self._registered_password,
                "full_name": "Regression Tester",
            },
        )
        # 201 on first registration; 409 if the email already exists from a
        # previous run (in-memory SQLite is re-used within the same process).
        assert resp.status_code in (201, 409), f"Unexpected status: {resp.status_code} — {resp.text}"
        if resp.status_code == 201:
            data = resp.json()
            assert "access_token" in data
            assert data["email"] == self._registered_email
            TestAuthEndpoints._token = data["access_token"]

    def test_login_existing_user(self, auth_client):
        resp = auth_client.post(
            "/auth/login",
            json={"email": self._registered_email, "password": self._registered_password},
        )
        assert resp.status_code == 200, f"Login failed: {resp.text}"
        data = resp.json()
        assert "access_token" in data
        TestAuthEndpoints._token = data["access_token"]

    def test_login_wrong_password_rejected(self, auth_client):
        resp = auth_client.post(
            "/auth/login",
            json={"email": self._registered_email, "password": "WrongPassword!"},
        )
        assert resp.status_code == 401

    def test_me_endpoint_with_valid_token(self, auth_client):
        if not TestAuthEndpoints._token:
            pytest.skip("Token not available — login test must run first")
        resp = auth_client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {TestAuthEndpoints._token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == self._registered_email

    def test_me_endpoint_without_token_rejected(self, auth_client):
        resp = auth_client.get("/auth/me")
        assert resp.status_code == 401

    def test_verify_token_valid(self, auth_client):
        if not TestAuthEndpoints._token:
            pytest.skip("Token not available")
        resp = auth_client.post(
            "/auth/verify-token",
            headers={"Authorization": f"Bearer {TestAuthEndpoints._token}"},
        )
        assert resp.status_code == 200
        assert resp.json().get("valid") is True

    def test_verify_token_tampered_rejected(self, auth_client):
        resp = auth_client.post(
            "/auth/verify-token",
            headers={"Authorization": "Bearer this.is.not.a.valid.jwt"},
        )
        assert resp.status_code == 401

    def test_register_weak_password_rejected(self, auth_client):
        resp = auth_client.post(
            "/auth/register",
            json={"email": "weakpw@example.com", "password": "short"},
        )
        assert resp.status_code == 422  # Pydantic validation error

    def test_register_invalid_email_rejected(self, auth_client):
        resp = auth_client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "ValidPassword123"},
        )
        assert resp.status_code == 422
