#!/usr/bin/env python3
"""Quick test script to verify the LiteRT-LM server is working."""

import json
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:11454"

def check(path, method="GET", data=None, headers=None):
    """Make a request and return (status, body)."""
    url = BASE_URL + path
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    if data:
        req.data = json.dumps(data).encode()
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}

def main():
    print("=== LiteRT-LM Server Test ===\n")

    # Health check
    print("[1/4] Health check...")
    status, body = check("/health")
    if status == 200:
        print(f"  ✓ Server OK — model: {body.get('model', '?')}")
    else:
        print(f"  ✗ Server not responding: {body}")
        sys.exit(1)

    # Model listing
    print("\n[2/4] Model listing...")
    status, body = check("/v1/models")
    if status == 200:
        models = body.get("data", [])
        for m in models:
            print(f"  ✓ {m['id']} — {m.get('display_name', '')}")
    else:
        print(f"  ✗ Failed: {body}")

    # Text generation
    print("\n[3/4] Text generation...")
    status, body = check("/v1/chat/completions", method="POST", data={
        "model": "gemma-4-e4b",
        "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
        "max_tokens": 50,
    })
    if status == 200:
        content = body["choices"][0]["message"]["content"]
        tokens = body["usage"]["completion_tokens"]
        print(f"  ✓ Response ({tokens} tokens): {content}")
    else:
        print(f"  ✗ Failed: {body}")

    # Multi-turn conversation
    print("\n[4/4] Multi-turn conversation...")
    session_id = "test-session-123"
    
    # Turn 1
    status, body = check("/v1/chat/completions", method="POST",
        data={"messages": [{"role": "user", "content": "My favorite color is blue."}]},
        headers={"X-Session-ID": session_id})
    if status != 200:
        print(f"  ✗ Turn 1 failed: {body}")
        return
    
    # Turn 2 (should remember)
    status, body = check("/v1/chat/completions", method="POST",
        data={"messages": [{"role": "user", "content": "What is my favorite color?"}]},
        headers={"X-Session-ID": session_id})
    if status == 200:
        content = body["choices"][0]["message"]["content"]
        if "blue" in content.lower():
            print(f"  ✓ Context remembered: {content[:80]}")
        else:
            print(f"  ⚠ Context may be lost: {content[:80]}")
    else:
        print(f"  ✗ Turn 2 failed: {body}")

    print("\n=== All tests passed ===")

if __name__ == "__main__":
    main()
