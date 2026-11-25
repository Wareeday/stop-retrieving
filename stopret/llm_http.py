"""HTTP helper for LLM APIs (Groq, Cerebras, Mistral, OpenRouter, Gemini, Ollama, ...).

Standard library only, and no imports from the rest of the package, so this single file can be
copied into another project (for example vtqa) and used as it is:

    from llm_http import post_json, AuthBlocked, RateLimitExhausted, ApiError
    reply = post_json(url, payload, {"Authorization": f"Bearer {key}"}, timeout=60)

What it does
------------
* Sends a browser-like User-Agent. Python's default ("Python-urllib/3.x", or "python-requests/2.x"
  for the requests library) is rejected by Cloudflare in front of some APIs with
  "403 - error code: 1010 (browser signature banned)". Override it with the `user_agent`
  argument or the STOPRET_USER_AGENT environment variable.
* 429, 408 and 5xx answers, dropped connections and timeouts are retried, waiting as long as the
  server asks (Retry-After header, or retryDelay in a Google-style error body) and otherwise with
  exponential backoff. A wait longer than `max_wait` seconds, or a 429 that never goes away, raises
  RateLimitExhausted instead of sleeping for hours.
* 401 and 403 are never retried. They raise AuthBlocked immediately, with the message
  "STOP: API Key invalid or blocked by Cloudflare (1010)" and a hint for the exact Cloudflare
  error code if the server sent one. The caller must not continue with the next item.
* Every other failure raises ApiError with a plain explanation (wrong model id, request too
  large, not JSON, certificate problem, ...). Nothing here ever prints or stores the API key.
"""
from __future__ import annotations

import email.utils
import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36")
USER_AGENT_ENV = "STOPRET_USER_AGENT"
STOP_HEADLINE = "STOP: API Key invalid or blocked by Cloudflare (1010)"


class ApiError(RuntimeError):
    """A request failed in a way that retrying will not fix (or retries ran out)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class AuthBlocked(ApiError):
    """HTTP 401 or 403: the key is rejected, or the request is blocked (e.g. Cloudflare 1010).
    Stop the whole run; do not try the next question."""


class RateLimitExhausted(ApiError):
    """The server keeps answering 429, or asked us to wait longer than we are willing to (for
    example a daily quota). Progress is not lost: run the same thing again later."""


# ---------------------------------------------------------------- small helpers

def resolve_user_agent(user_agent: str | None = None) -> str:
    """Explicit argument, else the STOPRET_USER_AGENT variable, else the browser-like default."""
    value = (user_agent or os.environ.get(USER_AGENT_ENV) or DEFAULT_USER_AGENT).strip()
    if not value or any(not 32 <= ord(ch) <= 126 for ch in value):
        raise ValueError("the User-Agent must be non-empty printable ASCII (no line breaks)")
    return value


def clean_api_key(raw: str | None, name: str) -> str:
    """Strip the whitespace, line breaks and quotes that sneak in when a key is copy-pasted or
    exported, and refuse keys that still contain impossible characters. The key itself is never
    echoed in an error message."""
    if raw is None:
        raise RuntimeError(f"set the {name} environment variable")
    key = raw.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    if not key:
        raise RuntimeError(f"{name} is set but empty")
    bad = [i for i, ch in enumerate(key) if not 33 <= ord(ch) <= 126]
    if bad:
        raise RuntimeError(
            f"{name} contains a space, line break or non-ASCII character (at position "
            f"{bad[0] + 1}). Copy the key again from the provider's console, without extra characters.")
    return key


def _safe_url(url: str) -> str:
    """Scheme, host and path only: a query string could carry a key."""
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


_SECRET_HEADERS = {"authorization", "x-goog-api-key", "x-api-key", "api-key"}


def _secrets_in(headers: dict) -> list[str]:
    """The credential values that are being sent, so they can be blanked out of any text that is
    shown to the user (some servers echo the key back in their error message)."""
    found = []
    for name, value in headers.items():
        if name.lower() in _SECRET_HEADERS and value:
            token = str(value).split()[-1]
            if len(token) >= 6:
                found.append(token)
    return found


def _snippet(raw: bytes, limit: int = 200, secrets: list[str] | None = None) -> str:
    text = " ".join(raw.decode("utf-8", "replace").split())
    for secret in secrets or []:
        text = text.replace(secret, "***")
    return text if len(text) <= limit else text[:limit] + "..."


def _read_quietly(err: urllib.error.HTTPError) -> bytes:
    try:
        return err.read()
    except (OSError, http.client.HTTPException):
        return b""


def _cloudflare_code(text: str) -> str | None:
    """Cloudflare's plain-text answer is 'error code: 1010'; its HTML page says 'Error 1010' and
    mentions the browser's signature."""
    match = re.search(r"(?:error(?:\s+code)?\s*:?\s*|cf-error-code[^0-9]*)(\d{4})", text,
                      re.IGNORECASE)
    if match:
        return match.group(1)
    lowered = text.lower()
    if "browser" in lowered and "signature" in lowered:
        return "1010"
    return None


_CLOUDFLARE_HINTS = {
    "1010": ("Cloudflare error 1010 means the request was rejected because of its browser "
             "signature (User-Agent). This client already sends a browser-like User-Agent, so if "
             "you set --user-agent or STOPRET_USER_AGENT yourself, remove it. Also try another "
             "network, switch off any VPN or proxy, and run `python -m stopret check --backend "
             "<name>` to test the connection with one tiny request."),
    "1015": "Cloudflare is rate limiting your IP address. Wait a while, then run the same command again.",
    "1020": ("A Cloudflare firewall rule denied access (this depends on your network or IP). "
             "Try another network or switch off the VPN/proxy."),
}
for _code in ("1006", "1007", "1008"):
    _CLOUDFLARE_HINTS[_code] = ("Cloudflare has banned your IP address or region for this site. "
                                "Try another network.")


def _blocked(status: int, shown_url: str, raw: bytes, secrets: list[str]) -> AuthBlocked:
    text = _snippet(raw, secrets=secrets)
    code = _cloudflare_code(text)
    if code in _CLOUDFLARE_HINTS:
        hint = _CLOUDFLARE_HINTS[code]
    elif status == 401:
        hint = ("The key was rejected. Check that it is copied completely, is active in the "
                "provider's console, and belongs to this provider (a key for another provider "
                "does not work).")
    else:
        hint = ("The server refused access. Check the key's permissions, that the model is "
                "enabled for your account and region, and whether a VPN/proxy is in the way.")
    return AuthBlocked(
        f"{STOP_HEADLINE}\n"
        f"  HTTP {status} from {shown_url}\n"
        f"  server said: {text or '(empty answer)'}\n"
        f"  {hint}\n"
        "  Nothing is lost: answers already received are saved, and the question that failed "
        "will be asked again the next time you run the same command.", status)


_CLIENT_HINTS = {
    400: ("The request was not accepted: check the model id and any options "
          "(--extra-body, --thinking, --max-tokens) against what this model supports."),
    404: ("Not found: the model id or the base URL is wrong. `python -m stopret check "
          "--backend <name>` lists the model ids your key can use."),
    413: ("The request is too large for this model or plan (often the tokens-per-minute limit). "
          "Use a model or plan with a larger limit."),
    422: "The request was understood but not accepted: check the model id and the options.",
}


def _client_error(status: int, shown_url: str, raw: bytes, secrets: list[str]) -> ApiError:
    hint = _CLIENT_HINTS.get(status, "The server rejected the request.")
    shown = _snippet(raw, secrets=secrets) or "(empty answer)"
    return ApiError(f"HTTP {status} from {shown_url}: {shown}\n  {hint}", status)


def _decode_json(raw: bytes, shown_url: str, secrets: list[str]) -> dict:
    try:
        data = json.loads(raw)
    except ValueError as err:
        raise ApiError(
            f"{shown_url} answered HTTP 200 but not with JSON (first characters: "
            f"{_snippet(raw, 120, secrets)!r}). A proxy, a captive portal or a Cloudflare challenge page "
            "can do this.") from err
    if not isinstance(data, dict):
        raise ApiError(f"{shown_url} answered with JSON that is not an object: {_snippet(raw, 120, secrets)!r}")
    if isinstance(data.get("error"), (dict, str)):
        raise ApiError(f"{shown_url} reported an error: {_snippet(raw, secrets=secrets)}")
    return data


def _retry_after_seconds(value: str | None) -> float | None:
    """Retry-After is either a number of seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        moment = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, moment.timestamp() - time.time())


def _retry_delay_from_body(raw: bytes) -> float | None:
    """Google-style 429 bodies carry the wait in error.details[].retryDelay, e.g. '34s'."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    for detail in details or []:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        match = re.fullmatch(r"([0-9.]+)s", delay or "")
        if match:
            return float(match.group(1))
    return None


# ---------------------------------------------------------------- the request

def request_json(method: str, url: str, payload: dict | None = None,
                 headers: dict | None = None, timeout: float = 300.0, *, max_retries: int = 8,
                 max_wait: float = 120.0, sleep=time.sleep, user_agent: str | None = None) -> dict:
    """Send one request and return the decoded JSON object (see the module docstring)."""
    request_headers = {"User-Agent": resolve_user_agent(user_agent), "Accept": "application/json",
                       **(headers or {})}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        request_headers.setdefault("Content-Type", "application/json")
    shown = _safe_url(url)
    secrets = _secrets_in(request_headers)

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return _decode_json(raw, shown, secrets)
        except urllib.error.HTTPError as err:
            raw = _read_quietly(err)
            status = err.code
            if status in (401, 403):
                raise _blocked(status, shown, raw, secrets) from err
            if not (status in (408, 429) or status >= 500):
                raise _client_error(status, shown, raw, secrets) from err
            if attempt == max_retries:
                if status == 429:
                    raise RateLimitExhausted(
                        f"still rate limited after {attempt} retries. Progress is cached; run the "
                        "same command again later and it will resume.", status) from err
                raise ApiError(f"HTTP {status} from {shown} even after {attempt} retries: "
                               f"{_snippet(raw, secrets=secrets) or '(empty answer)'}", status) from err
            wait = _retry_after_seconds(err.headers.get("Retry-After"))
            if wait is None:
                wait = _retry_delay_from_body(raw)
            if wait is None:
                wait = min(2.0 ** attempt, 60.0)
            if wait > max_wait:
                raise RateLimitExhausted(
                    f"server asked to wait {wait:.0f}s (limit reached). Progress is cached; run "
                    "the same command again later and it will resume.", status) from err
        except ssl.SSLCertVerificationError as err:
            raise _certificate_error(shown, err) from err
        except urllib.error.URLError as err:
            if isinstance(err.reason, ssl.SSLCertVerificationError):
                raise _certificate_error(shown, err.reason) from err
            if attempt == max_retries:
                raise ApiError(f"could not reach {shown} after {attempt} retries: {err.reason}") from err
            wait = min(2.0 ** attempt, 60.0)
        except (OSError, http.client.HTTPException) as err:
            # read timeouts, connection resets, dropped connections, incomplete reads, TLS errors
            if attempt == max_retries:
                raise ApiError(f"connection problem with {shown} after {attempt} retries: "
                               f"{type(err).__name__}: {err}") from err
            wait = min(2.0 ** attempt, 60.0)
        sleep(wait)
    raise AssertionError("unreachable")


def _certificate_error(shown_url: str, err: Exception) -> ApiError:
    return ApiError(
        f"the TLS certificate of {shown_url} could not be verified ({err}). Retrying will not "
        "help. Check the system date and time, and whether a VPN, proxy or antivirus is "
        "intercepting HTTPS traffic.")


def post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 300.0,
              **options) -> dict:
    return request_json("POST", url, payload, headers, timeout, **options)


def get_json(url: str, headers: dict | None = None, timeout: float = 60.0, **options) -> dict:
    return request_json("GET", url, None, headers, timeout, **options)
