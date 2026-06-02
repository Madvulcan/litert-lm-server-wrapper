#!/usr/bin/env python3
"""
Image Gallery Generator with LiteRT-LM Captions.

Usage:
    python3 gallery.py --input-dir /path/to/images --output gallery.html [--url http://127.0.0.1:11454] [--resize 800]
"""

import os
import sys
import json
import base64
import argparse
import urllib.request
import time
from pathlib import Path

API_URL = "http://127.0.0.1:11454/v1/chat/completions"
MAX_DIMENSION = 800
QUALITY = 85
PROMPT = "Describe this image concisely in one sentence."

def find_api_url(url):
    """Try to find a working API URL."""
    candidates = [
        url or API_URL,
        "http://127.0.0.1:11454/v1/chat/completions",
        "http://192.168.0.202:11454/v1/chat/completions",
    ]
    for u in candidates:
        try:
            host_port = u.split("/v1/")[0]
            req = urllib.request.urljoin(host_port + "/", "/health")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                print(f"[✓] Server found at {host_port}")
                return u
        except:
            continue
    return url or API_URL

def resize_image(path, max_dim=None):
    """Resize image before sending to API. Set max_dim=0 to disable."""
    if max_dim is None:
        max_dim = MAX_DIMENSION
    from PIL import Image
    import io
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max_dim > 0 and max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=QUALITY)
    return buf.getvalue()

def caption_image(path, api_url, session_id, resize):
    """Get a caption from the LiteRT-LM API."""
    try:
        with open(path, "rb") as f:
            orig_data = f.read()
        if resize > 0:
            img_data = resize_image(path, resize)
        else:
            img_data = orig_data
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        img_b64 = base64.b64encode(img_data).decode()
        body = json.dumps({
            "model": "gemma-4-e4b",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": PROMPT}
            ]}],
            "max_tokens": 128,
            "temperature": 0.1,
        }).encode()
        req = urllib.request.Request(api_url, data=body,
            headers={"Content-Type": "application/json", "X-Session-ID": session_id}, method="POST")
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

captions = []

def handler(path, caption):
    """Collect captions for the gallery."""
    captions.append((Path(path).name, caption))

def generate_html(data, img_dir, api_url, title, total_time, num_images):
    """Generate an HTML gallery."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
    h1 {{ text-align: center; margin: 30px 0; color: #e94560; font-size: 2.5em; }}
    .stats {{ text-align: center; margin: 20px 0; color: #666; font-size: 0.9em; }}
    .gallery {{ max-width: 900px; margin: 0 auto; }}
    .entry {{ margin-bottom: 40px; background: #16213e; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
    .entry img {{ width: 100%; height: auto; display: block; }}
    .caption {{ padding: 16px 20px; font-size: 1.05em; line-height: 1.6; color: #c0c0c0; }}
    .filename {{ font-size: 0.8em; color: #666; padding: 8px 20px; background: #0f3460; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="stats">{num_images} images · {total_time:.1f}s ({total_time/num_images:.2f}s/image) · {api_url}</div>
<div class="gallery">
"""
    for filename, caption in data:
        html += f'<div class="entry"><img src="{filename}" alt="{caption[:80]}"><div class="filename">{filename}</div><div class="caption">{caption}</div></div>\n'
    html += "</div></body></html>"
    return html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an HTML gallery with LiteRT-LM image captions.")
    parser.add_argument("--input-dir", required=True, help="Directory containing images")
    parser.add_argument("--output", default="gallery.html", help="Output HTML file path")
    parser.add_argument("--url", default=None, help="LiteRT-LM API URL")
    parser.add_argument("--resize", type=int, default=800, help="Max dimension for image resize (0=disable)")
    parser.add_argument("--title", default="Image Gallery", help="Gallery title")
    args = parser.parse_args()

    api_url = find_api_url(args.url)
    img_dir = Path(args.input_dir)
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])

    if not images:
        print(f"No images found in {img_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images")
    print(f"API: {api_url}")
    print(f"Resize: {args.resize if args.resize > 0 else 'disabled'}")
    print()

    results = []
    start = time.time()
    for i, img_path in enumerate(images):
        session_id = f"gallery-{i}"
        print(f"[{i+1}/{len(images)}] {img_path.name}...", end=" ", flush=True)
        t0 = time.time()
        cap = caption_image(str(img_path), api_url, session_id, args.resize)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s) {cap[:60]}...")
        results.append((img_path.name, cap))

    total = time.time() - start
    print(f"\nTotal: {total:.1f}s ({total/len(results):.2f}s per image)")

    html = generate_html(results, img_dir, api_url, args.title, total, len(results))
    output = Path(args.output)
    with open(output, "w") as f:
        f.write(html)
    print(f"Gallery saved to: {output.absolute()}")
