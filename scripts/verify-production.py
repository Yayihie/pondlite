#!/usr/bin/env python3
"""Verify the deployed PondLite product without printing credentials."""

import argparse
import base64
import http.cookiejar
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def load_credentials():
    values = dict(os.environ)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values.setdefault(key, value)
    return values["PONDLITE_USERNAME"], values["PONDLITE_PASSWORD"]


def request_json(url, username, password, payload):
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {credentials}"},
    )
    return json.loads(urllib.request.urlopen(request, timeout=90).read())


def csrf(page):
    match = re.search(r'name="_csrf_token" value="([^"]+)"', page)
    assert match, "CSRF token missing from page"
    return match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1")
    parser.add_argument("--eval", action="store_true", help="also run the browser evaluation flow")
    args = parser.parse_args()
    username, password = load_credentials()
    api = args.base + ":8420"
    ui = args.base + ":8421"
    anonymous_error = None
    try:
        request_json(api + "/v1/chat/completions", "wrong", "wrong", {
            "messages": [{"role": "user", "content": "How do I cancel?"}],
        })
    except urllib.error.HTTPError as exc:
        anonymous_error = exc.code
    assert anonymous_error == 401, anonymous_error

    known = request_json(api + "/v1/chat/completions", username, password, {
        "messages": [{"role": "user", "content": "How do I cancel my subscription?"}],
    })
    unknown = request_json(api + "/v1/chat/completions", username, password, {
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    })
    assert known["pondlite"]["mode"] == "grounded"
    assert "Settings > Billing > Manage plan > Cancel" in known["choices"][0]["message"]["content"]
    assert unknown["pondlite"]["mode"] == "abstained"

    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    login_page = opener.open(ui + "/login", timeout=15).read().decode()
    response = opener.open(ui + "/login", data=urllib.parse.urlencode({
        "_csrf_token": csrf(login_page), "username": username, "password": password, "next": "/ingest",
    }).encode(), timeout=15)
    assert response.status == 200
    assert "Start with real conversations" in response.read().decode()

    report = {
        "anonymous_api_status": anonymous_error,
        "ui_login": True,
        "known_mode": known["pondlite"]["mode"],
        "known_confidence": known["pondlite"]["confidence"],
        "known_answer": known["choices"][0]["message"]["content"],
        "unknown_mode": unknown["pondlite"]["mode"],
    }

    if args.eval:
        eval_url = ui + "/eval"
        page = opener.open(eval_url, timeout=15).read().decode()
        run_dir = re.search(r'name="run_dir"[^>]*>\s*<option value="([^"]+)"', page).group(1)
        response = opener.open(eval_url, data=urllib.parse.urlencode({
            "_csrf_token": csrf(page), "run_dir": run_dir,
        }).encode(), timeout=15)
        assert response.status == 200
        deadline = time.time() + 300
        while time.time() < deadline:
            page = opener.open(eval_url + "?run=" + urllib.parse.quote(run_dir), timeout=15).read().decode()
            if "Evaluation failed" in page:
                raise AssertionError("browser evaluation failed")
            if "Grounded policy match" in page and "8/8" in page:
                report["one_click_eval"] = "done"
                report["grounded_policy_matches"] = "8/8"
                break
            time.sleep(2)
        else:
            raise TimeoutError("browser evaluation did not complete in 300 seconds")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
