import time
from typing import Any, Dict, Optional

import requests


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def post_with_retries(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
    max_attempts: int = 4,
    backoff_seconds: float = 2.0,
) -> requests.Response:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=json, timeout=timeout)
            if resp.status_code not in RETRYABLE_STATUS_CODES:
                resp.raise_for_status()
                return resp
            last_error = requests.HTTPError(
                f"{resp.status_code} Server Error for url: {url}",
                response=resp,
            )
        except requests.RequestException as exc:
            last_error = exc

        if attempt == max_attempts:
            break

        retry_after = None
        if isinstance(last_error, requests.HTTPError) and last_error.response is not None:
            retry_after = last_error.response.headers.get('Retry-After')

        if retry_after is not None:
            try:
                sleep_for = max(float(retry_after), backoff_seconds * attempt)
            except ValueError:
                sleep_for = backoff_seconds * attempt
        else:
            sleep_for = backoff_seconds * attempt
        time.sleep(sleep_for)

    if last_error:
        raise last_error

    raise RuntimeError(f"Request failed without a captured error for {url}")
