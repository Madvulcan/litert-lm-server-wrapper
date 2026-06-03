#!/usr/bin/env python3
"""
LiteRT-LM Server Benchmark Suite
Tests: vision batch, multi-turn, concurrent, streaming, long-form, compression
"""

import json
import time
import base64
import urllib.request
import urllib.error
import statistics
import threading
from pathlib import Path

API_URL = "http://192.168.0.202:11454/v1/chat/completions"
HEALTH_URL = "http://192.168.0.202:11454/health"
MODEL = "gemma-4-e4b"

def post(data, headers=None, timeout=120):
    """Send a POST request and return (status, body, elapsed)."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(API_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        elapsed = time.time() - t0
        return resp.status, json.loads(resp.read()), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        return e.code, json.loads(e.read()), elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return 0, {"error": str(e)}, elapsed

def check_health():
    """Check server health."""
    try:
        req = urllib.request.Request(HEALTH_URL)
        resp = urllib.request.urlopen(req, timeout=5)
        return json.loads(resp.read())
    except:
        return None

def stat_summary(values):
    """Return min, max, median, mean, p95."""
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": s[0],
        "max": s[-1],
        "median": statistics.median(s),
        "mean": statistics.mean(s),
        "p95": s[int(len(s) * 0.95)] if len(s) > 1 else s[0],
    }

# ═══════════════════════════════════════════════════════════════
# TEST 1: Vision Stress Test — 112 images, per-image latency
# ═══════════════════════════════════════════════════════════════
def test_vision_batch():
    print("\n" + "="*60)
    print("TEST 1: Vision Stress Test (112 images)")
    print("="*60)

    img_dir = Path.home() / "Pictures/hookorbycrook"
    extensions = {".png", ".jpg", ".jpeg"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])

    if not images:
        print("  ✗ No images found, skipping")
        return {}

    print(f"  Found {len(images)} images")

    latencies = []
    errors = 0
    start = time.time()

    for i, img_path in enumerate(images):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        status, body, elapsed = post({
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "Describe this image concisely in one sentence."}
            ]}],
            "max_tokens": 128,
        }, headers={"X-Session-ID": f"vision-{i}"})

        if status == 200:
            latencies.append(elapsed)
        else:
            errors += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(images)}] {elapsed:.2f}s last, {len(latencies)} ok, {errors} errors")

    total = time.time() - start
    stats = stat_summary(latencies)

    print(f"\n  Results:")
    print(f"    Total time:     {total:.1f}s")
    print(f"    Images OK:      {len(latencies)}/{len(images)}")
    print(f"    Errors:         {errors}")
    print(f"    Min latency:    {stats['min']:.2f}s")
    print(f"    Max latency:    {stats['max']:.2f}s")
    print(f"    Median latency: {stats['median']:.2f}s")
    print(f"    Mean latency:   {stats['mean']:.2f}s")
    print(f"    P95 latency:    {stats['p95']:.2f}s")
    print(f"    Throughput:     {len(latencies)/total:.1f} img/s")

    return {"test": "vision_batch", "stats": stats, "total": total, "ok": len(latencies), "errors": errors}

# ═══════════════════════════════════════════════════════════════
# TEST 2: Multi-Turn Conversation (20 turns)
# ═══════════════════════════════════════════════════════════════
def test_multi_turn():
    print("\n" + "="*60)
    print("TEST 2: Multi-Turn Conversation (20 turns)")
    print("="*60)

    session_id = "multi-turn-benchmark"
    messages = []
    latencies = []

    prompts = [
        "My name is Alice and I live in Paris.",
        "What is my name?",
        "What city do I live in?",
        "I have a cat named Whiskers. What is my cat's name?",
        "I work as a software engineer. What is my job?",
        "My favorite programming language is Python. What is my favorite language?",
        "I enjoy hiking on weekends. What do I enjoy doing?",
        "I have a brother named Bob. What is my brother's name?",
        "I drive a red car. What color is my car?",
        "I studied at MIT. Where did I study?",
        "I speak English and French. What languages do I speak?",
        "I am 32 years old. How old am I?",
        "I have a garden with roses. What flowers are in my garden?",
        "I prefer tea over coffee. What do I prefer?",
        "I play the piano. What instrument do I play?",
        "I visited Japan last year. Where did I visit?",
        "I am allergic to peanuts. What am I allergic to?",
        "I have a bicycle. What vehicle do I own besides a car?",
        "I like to read science fiction. What genre do I like?",
        "Summarize everything you know about me in 3 sentences.",
    ]

    for i, prompt in enumerate(prompts):
        messages.append({"role": "user", "content": prompt})
        status, body, elapsed = post({
            "model": MODEL,
            "messages": messages,
            "max_tokens": 128,
        }, headers={"X-Session-ID": session_id})

        if status == 200:
            latencies.append(elapsed)
            # Add assistant response to message history
            assistant_msg = body["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": assistant_msg})
        else:
            print(f"  ✗ Turn {i+1} failed: {body}")
            break

        print(f"  Turn {i+1:2d}: {elapsed:.2f}s — {body['choices'][0]['message']['content'][:60]}...")

    stats = stat_summary(latencies)
    print(f"\n  Results:")
    print(f"    Turns completed: {len(latencies)}/{len(prompts)}")
    print(f"    Min latency:     {stats['min']:.2f}s")
    print(f"    Max latency:     {stats['max']:.2f}s")
    print(f"    Median latency:  {stats['median']:.2f}s")
    print(f"    Mean latency:    {stats['mean']:.2f}s")

    # Check if the final summary is coherent
    final_response = messages[-1]["content"] if messages else ""
    context_score = sum(1 for kw in ["Alice", "Paris", "Whiskers", "Python", "32"] if kw in final_response)
    print(f"    Context score:   {context_score}/5 key facts in final summary")

    return {"test": "multi_turn", "stats": stats, "turns": len(latencies), "context_score": context_score}

# ═══════════════════════════════════════════════════════════════
# TEST 3: Concurrent Sessions (5 simultaneous)
# ═══════════════════════════════════════════════════════════════
def test_concurrent():
    print("\n" + "="*60)
    print("TEST 3: Concurrent Sessions (5 simultaneous)")
    print("="*60)

    prompts = [
        "What is the capital of France?",
        "What is 17 × 23?",
        "Name three colors.",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
    ]

    results = [None] * len(prompts)
    errors = []

    def worker(idx, prompt):
        status, body, elapsed = post({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
        }, headers={"X-Session-ID": f"concurrent-{idx}"})
        results[idx] = {"status": status, "body": body, "elapsed": elapsed, "prompt": prompt}

    threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    total = time.time() - start

    ok = sum(1 for r in results if r and r["status"] == 200)
    latencies = [r["elapsed"] for r in results if r and r["status"] == 200]

    for i, r in enumerate(results):
        if r:
            content = r["body"].get("choices", [{}])[0].get("message", {}).get("content", "")[:50] if r["status"] == 200 else str(r["body"])[:50]
            print(f"  Session {i+1}: {r['elapsed']:.2f}s — {content}...")
        else:
            print(f"  Session {i+1}: TIMEOUT")

    stats = stat_summary(latencies)
    print(f"\n  Results:")
    print(f"    Completed:    {ok}/{len(prompts)}")
    print(f"    Total wall:   {total:.2f}s")
    print(f"    Min latency:  {stats['min']:.2f}s")
    print(f"    Max latency:  {stats['max']:.2f}s")
    print(f"    Median:       {stats['median']:.2f}s")

    return {"test": "concurrent", "ok": ok, "total": len(prompts), "wall_time": total, "stats": stats}

# ═══════════════════════════════════════════════════════════════
# TEST 4: Streaming vs Non-Streaming
# ═══════════════════════════════════════════════════════════════
def test_streaming():
    print("\n" + "="*60)
    print("TEST 4: Streaming vs Non-Streaming")
    print("="*60)

    prompt = "Write a 200-word essay about the history of computing."

    # Non-streaming
    status, body, elapsed_ns = post({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": False,
    })
    tokens_ns = body.get("usage", {}).get("completion_tokens", 0) if status == 200 else 0
    print(f"  Non-stream: {elapsed_ns:.2f}s, {tokens_ns} tokens")

    # Streaming — measure time to first chunk and total
    import io
    body_s = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": True,
    }).encode()
    req = urllib.request.Request(API_URL, data=body_s,
        headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.time()
    first_chunk_time = None
    chunks = 0
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                if first_chunk_time is None:
                    first_chunk_time = time.time() - t0
                chunks += 1
        elapsed_s = time.time() - t0
    except Exception as e:
        elapsed_s = time.time() - t0
        print(f"  ✗ Stream error: {e}")

    print(f"  Stream:      {elapsed_s:.2f}s, {chunks} chunks, first chunk at {first_chunk_time:.2f}s" if first_chunk_time else f"  Stream: {elapsed_s:.2f}s, no chunks received")

    return {
        "test": "streaming",
        "non_stream_time": elapsed_ns,
        "non_stream_tokens": tokens_ns,
        "stream_time": elapsed_s,
        "stream_chunks": chunks,
        "first_chunk_time": first_chunk_time,
    }

# ═══════════════════════════════════════════════════════════════
# TEST 5: Long-Form Generation (1000 tokens)
# ═══════════════════════════════════════════════════════════════
def test_long_form():
    print("\n" + "="*60)
    print("TEST 5: Long-Form Generation (1000 tokens)")
    print("="*60)

    prompt = """Write a detailed 800-word essay covering the history of artificial intelligence. 
Include: the Dartmouth Conference, expert systems, the AI winter, the rise of machine learning, 
deep learning breakthroughs, and current large language models. Use proper section headings."""

    status, body, elapsed = post({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }, timeout=180)

    if status == 200:
        tokens = body.get("usage", {}).get("completion_tokens", 0)
        content = body["choices"][0]["message"]["content"]
        words = len(content.split())
        tok_s = tokens / elapsed if elapsed > 0 else 0

        print(f"  Time:          {elapsed:.2f}s")
        print(f"  Tokens:        {tokens}")
        print(f"  Words:         {words}")
        print(f"  Speed:         {tok_s:.1f} tok/s")
        print(f"  Content hash:  {hash(content) % 100000:05d}")
        print(f"  First 100 chars: {content[:100]}...")

        return {"test": "long_form", "time": elapsed, "tokens": tokens, "words": words, "tok_s": tok_s}
    else:
        print(f"  ✗ Failed: {body}")
        return {"test": "long_form", "error": str(body)}

# ═══════════════════════════════════════════════════════════════
# TEST 6: Compression (summarize long text)
# ═══════════════════════════════════════════════════════════════
def test_compression():
    print("\n" + "="*60)
    print("TEST 6: Compression (summarize long text)")
    print("="*60)

    # Generate a long text to compress
    long_text = """
    Artificial intelligence (AI) has undergone remarkable evolution since its inception in the 1950s. 
    The field began with the Dartmouth Conference in 1956, where John McCarthy and his colleagues 
    coined the term "artificial intelligence" and laid the groundwork for decades of research.
    
    In the early decades, AI research focused on symbolic reasoning and expert systems. These 
    rule-based programs could perform specific tasks like medical diagnosis or chemical analysis, 
    but they lacked the ability to learn from data or generalize to new situations. The limitations 
    of these approaches became apparent during the "AI winters" of the 1970s and 1980s, when funding 
    dried up as early promises went unfulfilled.
    
    The resurgence came with the rise of machine learning in the 1990s and 2000s. Rather than 
    programming explicit rules, researchers developed algorithms that could learn patterns from 
    data. Support vector machines, random forests, and other techniques proved effective for 
    many practical applications, from spam filtering to recommendation systems.
    
    The deep learning revolution, beginning around 2012, transformed the field once again. 
    Neural networks with many layers could learn hierarchical representations of data, achieving 
    breakthrough results in image recognition, speech processing, and natural language understanding. 
    The availability of large datasets and powerful GPUs accelerated progress dramatically.
    
    Today, large language models like GPT-4, Claude, and Gemma represent the cutting edge of AI. 
    These models, trained on vast corpora of text, can generate human-like text, answer questions, 
    write code, and assist with creative tasks. They have sparked both excitement about AI's 
    potential and concern about its implications for society, employment, and human cognition.
    
    The future of AI promises continued advancement in areas like multimodal understanding, 
    reasoning, and agentic systems. Researchers are working to make AI more efficient, 
    interpretable, and aligned with human values. The field remains one of the most dynamic 
    and consequential areas of computer science and technology.
    """ * 3  # Triple it for a longer text

    word_count = len(long_text.split())
    print(f"  Input: {word_count} words")

    status, body, elapsed = post({
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Summarize the following text in exactly 3 sentences:\n\n{long_text}"}],
        "max_tokens": 256,
    }, timeout=120)

    if status == 200:
        summary = body["choices"][0]["message"]["content"]
        summary_words = len(summary.split())
        ratio = summary_words / word_count * 100 if word_count > 0 else 0

        print(f"  Time:          {elapsed:.2f}s")
        print(f"  Input words:   {word_count}")
        print(f"  Output words:  {summary_words}")
        print(f"  Compression:   {ratio:.1f}% of original")
        print(f"  Summary:       {summary[:150]}...")

        return {"test": "compression", "time": elapsed, "input_words": word_count, "output_words": summary_words, "ratio": ratio}
    else:
        print(f"  ✗ Failed: {body}")
        return {"test": "compression", "error": str(body)}

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         LiteRT-LM Server Benchmark Suite                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Health check
    health = check_health()
    if not health:
        print("\n✗ Server not responding. Start it first:")
        print("  python3 litert-lm-server.py --mode server --port 11454 --model-size 4b")
        return

    print(f"\nServer: {health.get('display_name', '?')}")
    print(f"Model:  {health.get('model', '?')}")
    print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    results.append(test_vision_batch())
    results.append(test_multi_turn())
    results.append(test_concurrent())
    results.append(test_streaming())
    results.append(test_long_form())
    results.append(test_compression())

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for r in results:
        if "error" not in r:
            name = r.get("test", "?")
            if name == "vision_batch":
                print(f"  Vision batch:    {r['stats']['median']:.2f}s median, {r['ok']}/{r['ok']+r['errors']} ok")
            elif name == "multi_turn":
                print(f"  Multi-turn:      {r['stats']['median']:.2f}s median, {r['turns']} turns, context={r['context_score']}/5")
            elif name == "concurrent":
                print(f"  Concurrent:      {r['ok']}/{r['total']} ok, {r['wall_time']:.2f}s wall")
            elif name == "streaming":
                print(f"  Streaming:       {r['stream_chunks']} chunks, first at {r['first_chunk_time']:.2f}s" if r.get('first_chunk_time') else f"  Streaming: no chunks")
            elif name == "long_form":
                print(f"  Long-form:       {r['time']:.2f}s, {r['tokens']} tok, {r['tok_s']:.1f} tok/s")
            elif name == "compression":
                print(f"  Compression:     {r['time']:.2f}s, {r['input_words']}→{r['output_words']} words ({r['ratio']:.1f}%)")

    print("\nDone.")

if __name__ == "__main__":
    main()
