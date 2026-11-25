"""Drop-in pattern for a project that uses the requests library (for example vtqa)."""
import email.utils
import sys
import time

import requests

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
session = requests.Session()
session.headers.update({"User-Agent": BROWSER_UA, "Accept": "application/json"})


def _wait_seconds(response, attempt):
    value = response.headers.get("Retry-After")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:                                              # Retry-After can also be an HTTP date
        return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return min(2.0 ** attempt, 60.0)


def call_llm(url, payload, key, tries=6, timeout=120):
    """Returns the decoded JSON. Stops the whole program on 401/403 (SystemExit is not caught by
    `except Exception`, so a per-question try/except will not swallow it)."""
    for attempt in range(tries):
        try:
            response = session.post(url, json=payload, timeout=timeout,
                                    headers={"Authorization": f"Bearer {key}"})
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(2.0 ** attempt, 60.0))
            continue
        if response.status_code in (401, 403):
            sys.exit("STOP: API Key invalid or blocked by Cloudflare (1010)\n"
                     f"  HTTP {response.status_code}: {response.text[:200]}")
        if response.status_code in (408, 429) or response.status_code >= 500:
            time.sleep(_wait_seconds(response, attempt))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"the API kept failing after {tries} tries")
