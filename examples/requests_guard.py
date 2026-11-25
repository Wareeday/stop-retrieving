"""For projects that call an LLM API with the `requests` library (for example vtqa).

Copy `post_json_guarded` into the project and use it instead of `requests.post(...)`. It does
three things:

1. Sends a browser-like User-Agent. The default "python-requests/2.x" is rejected by Cloudflare in
   front of some APIs with "403 - error code: 1010".
2. Retries 429, 408 and 5xx answers, timeouts and dropped connections, waiting as long as the
   server asks (Retry-After) and otherwise with growing pauses.
3. Raises ApiStop immediately on 401 or 403. Catch it OUTSIDE the loop over questions, stop there,
   and do not write a result for the question that failed, so that the next run asks it again.

Typical use:

    results = load_results()                       # what is already finished
    for question in questions:
        if question.id in results:                 # only finished questions are skipped
            continue
        try:
            reply = post_json_guarded(URL, payload(question), api_key)
        except ApiStop as stop:
            save_results(results)                  # keep everything that did succeed
            sys.exit(str(stop))                    # stop the run, do not try the next question
        results[question.id] = parse(reply)        # recorded only after a successful reply
"""
import time

import requests

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"


class ApiStop(RuntimeError):
    """401/403: the key is invalid or the request is blocked. Do not continue with other items."""


def post_json_guarded(url, payload, api_key, timeout=60, max_retries=6, extra_headers=None,
                      sleep=time.sleep):
    headers = {"User-Agent": BROWSER_UA, "Accept": "application/json",
               "Authorization": f"Bearer {api_key.strip()}", **(extra_headers or {})}
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == max_retries:
                raise
            sleep(min(2 ** attempt, 60))
            continue
        if response.status_code in (401, 403):
            body = " ".join(response.text.split())[:200].replace(api_key.strip(), "***")
            raise ApiStop("STOP: API Key invalid or blocked by Cloudflare (1010)\n"
                          f"  HTTP {response.status_code}: {body}")
        if response.status_code in (408, 429) or response.status_code >= 500:
            if attempt == max_retries:
                response.raise_for_status()
            try:
                wait = float(response.headers.get("Retry-After", ""))
            except ValueError:
                wait = min(2 ** attempt, 60)
            sleep(min(wait, 120))
            continue
        response.raise_for_status()
        return response.json()
    raise AssertionError("unreachable")
