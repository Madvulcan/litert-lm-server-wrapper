# LiteRT-LM OpenAI-Compatible API Server

An OpenAI-compatible API server for Google's Gemma 4 E2B/E4B models using LiteRT-LM, with support for text, image (vision), and audio input. Includes conversation persistence, model auto-switching, and benchmarking.

## Features

- **OpenAI-compatible API** — Drop-in replacement for `https://api.openai.com/v1`
- **Vision + Audio** — Send images and audio alongside text in a single request
- **Multi-Token Prediction (MTP)** — Built-in speculative decoding for ~2× text generation speedup
- **Conversation persistence** — Maintains context across multi-turn chats via `X-Session-ID` header
- **Model auto-switching** — Automatically swaps between E2B and E4B based on the `model` field in requests
- **Ollama-compatible endpoints** — Also supports `/api/tags` and `/api/chat`
- **Low memory footprint** — E4B model uses ~7 GB GPU VRAM total (includes vision encoder + MTP drafter)

## Architecture

```
Client (Chatbox, OpenWebUI, etc.)
    │
    ▼
LiteRT-LM Server (port 11454)
    │
    ├── /v1/chat/completions  (OpenAI-compatible)
    ├── /chat/completions     (alias)
    ├── /api/chat             (Ollama-compatible)
    ├── /v1/models            (model listing)
    ├── /health               (health check)
    └── /reset                (clear conversation)
    │
    ▼
LiteRT-LM Engine (GPU)
    ├── Text decoder (Gemma 4)
    ├── Vision encoder (CLIP-based)
    ├── Audio encoder
    └── MTP drafter (built-in speculative decoding)
```

## Requirements

- **GPU**: NVIDIA with ≥8 GB VRAM (tested on RTX 4070 Super 12 GB)
- **OS**: Linux (Ubuntu 24.04 LTS recommended)
- **Python**: 3.10+
- **Model**: Gemma 4 E2B or E4B `.litertlm` file (~2.4 GB or ~3.5 GB)

## Quick Start

### 1. Install LiteRT-LM

```bash
pip install litert-lm
```

Or with a specific version:

```bash
pip install litert-lm==0.1.1
```

### 2. Download Models

Models are automatically downloaded from HuggingFace on first use, or you can pre-download them:

```bash
# Option A: Let the server auto-download (happens on first API call)
python3 litert-lm-server.py --mode server --port 11454 --model-size 4b

# Option B: Pre-download with the litert-lm CLI
litert-lm download litert-community/gemma-4-E4B-it-litert-lm
litert-lm download litert-community/gemma-4-E2B-it-litert-lm
```

**Model sources:**
| Model | HuggingFace Repo | Size | VRAM |
|---|---|---|---|
| Gemma 4 E2B | `litert-community/gemma-4-E2B-it-litert-lm` | 2.4 GB | ~4.6 GB |
| Gemma 4 E4B | `litert-community/gemma-4-E4B-it-litert-lm` | 3.5 GB | ~7.0 GB |

### 3. Configure Model Paths

The server looks for models in this order:

1. `~/llama-cpp-server/models/gemma-4-E4B-it.litertlm`
2. `~/llama-cpp-server/models/gemma-4-E2B-it.litertlm`
3. HuggingFace cache: `~/.cache/huggingface/hub/models--litert-community--gemma-4-E*-it-litert-lm/`

Symlink or copy your model files to `~/llama-cpp-server/models/`:

```bash
mkdir -p ~/llama-cpp-server/models
ln -s /path/to/gemma-4-E4B-it.litertlm ~/llama-cpp-server/models/
ln -s /path/to/gemma-4-E2B-it.litertlm ~/llama-cpp-server/models/
```

### 4. Start the Server

**Direct:**

```bash
python3 litert-lm-server.py --mode server --port 11454 --model-size 4b
```

**Systemd (recommended):**

```bash
cp litert-lm-gemma.service ~/.config/systemd/user/
sed -i "s|/path/to|$(pwd)|g" ~/.config/systemd/user/litert-lm-gemma.service
# Edit to change --model-size if needed

systemctl --user daemon-reload
systemctl --user enable --now litert-lm-gemma.service

# Check status
systemctl --user status litert-lm-gemma.service
journalctl --user -u litert-lm-gemma.service -f
```

## API Usage

All endpoints accept standard OpenAI-format requests. The default model is E2B; specify `"model": "gemma-4-e4b"` to use the larger model (or trigger auto-switching).

### Health Check

```bash
curl http://127.0.0.1:11454/health
# {"status": "ok", "model": "gemma-4-e4b", "display_name": "Gemma 4 E4B (LiteRT-LM)", "active_conversations": 0}
```

### Model Listing

```bash
curl http://127.0.0.1:11454/v1/models
```

### Text Chat

```bash
curl http://127.0.0.1:11454/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 256
  }'
```

### Image Captioning (Vision)

```bash
# Inline base64 image
IMAGE_B64=$(base64 -w0 image.png)
curl http://127.0.0.1:11454/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMAGE_B64"'"}},
      {"type": "text", "text": "Describe this image concisely in one sentence."}
    ]}],
    "max_tokens": 128
  }'
```

### Multi-Turn Conversation

Use the `X-Session-ID` header to maintain context across requests:

```bash
# Turn 1
curl http://127.0.0.1:11454/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-chat-123" \
  -d '{"messages": [{"role": "user", "content": "My name is Alice."}]}'

# Turn 2 (remembers context)
curl http://127.0.0.1:11454/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-chat-123" \
  -d '{"messages": [{"role": "user", "content": "What is my name?"}]}'
```

### Reset Conversation

```bash
curl -X POST http://127.0.0.1:11454/reset \
  -H "X-Session-ID: my-chat-123"
```

### Streaming

```bash
curl http://127.0.0.1:11454/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Count to 10"}], "stream": true}'
```

## Image Gallery Generator

The included `gallery.py` script generates an HTML gallery with AI captions for all images in a directory:

```bash
# Basic usage
python3 gallery.py --input-dir ~/Pictures/my-photos --output gallery.html

# With custom server URL and no image resizing
python3 gallery.py --input-dir ~/Pictures --output gallery.html \
  --url http://127.0.0.1:11454 --resize 0

# With custom title
python3 gallery.py --input-dir ~/Pictures --output gallery.html \
  --title "My Vacation Photos"
```

This will:
1. Scan the directory for `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` files
2. Send each image to the LiteRT-LM server for captioning
3. Generate a dark-themed HTML gallery with images and captions

## Configuration

### Server Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `cli` | `cli`, `server`, or `benchmark` |
| `--port` | `11454` | Server port |
| `--model-size` | `2b` | `2b` or `4b` |
| `--model-path` | None | Override auto-detection with explicit path |
| `--prompt` | `Hello` | CLI mode prompt |
| `--image` | None | CLI mode image path |
| `--audio` | None | CLI mode audio path |

### Environment Variables

The systemd service sets these automatically:

| Variable | Value | Purpose |
|---|---|---|
| `PATH` | `/usr/local/bin:/usr/bin:/bin` | Binary discovery |
| `HOME` | User home | Model file paths |
| `XDG_RUNTIME_DIR` | `/run/user/<uid>` | GPU device access |

### Conversation Settings (in code)

Edit these constants at the top of `litert-lm-server.py`:

| Constant | Default | Description |
|---|---|---|
| `MAX_CONVERSATIONS` | 100 | Max concurrent sessions |
| `CONVERSATION_TIMEOUT_SECS` | 600 | Idle timeout (10 min) |

## Performance

Benchmarked on NVIDIA RTX 4070 Super (12 GB VRAM), Gemma 4 E4B:

| Task | Speed | Notes |
|---|---|---|
| Text generation | ~157 tok/s | With MTP enabled |
| Image captioning | ~0.65s/image | Vision encoder is the bottleneck |
| Multi-turn chat | ~0.3s/turn | After initial context is built |

**MTP (Multi-Token Prediction)** is enabled by default via `enable_speculative_decoding=True`. This provides ~2× text generation speedup with no quality loss. The MTP drafter weights are built into the `.litertlm` file — no separate download needed.

## Troubleshooting

### Model not found

```
FileNotFoundError: Model not found for size '4b'
```

Ensure the model file exists at `~/llama-cpp-server/models/gemma-4-E4B-it.litertlm` or is in the HuggingFace cache.

### GPU out of memory

The E4B model needs ~7 GB VRAM. Close other GPU applications or switch to E2B:

```bash
python3 litert-lm-server.py --mode server --port 11454 --model-size 2b
```

### Server not responding

Check the log file:

```bash
tail -50 /tmp/litert-lm.log
```

### Deterministic output (no temperature sensitivity)

This is a known limitation of the current LiteRT-LM engine for Gemma 4. The model produces identical responses regardless of temperature/top_p/seed settings. This is likely a `.litertlm` conversion issue (greedy decoding only).

## Known Limitations

1. **Single-session engine** — Only one active conversation per engine instance. New sessions close previous ones.
2. **Deterministic output** — Temperature/top_p/seed are accepted but not honored by the C++ engine.
3. **No batching** — Each request is processed sequentially.
4. **Linux only** — Tested on Ubuntu 24.04 LTS. Windows/macOS not tested.

## License

Apache 2.0 (same as Gemma 4). The LiteRT-LM engine is subject to Google's Gemma license terms.

## Acknowledgments

- Google DeepMind for Gemma 4
- The LiteRT-LM team at Google
- HuggingFace for model hosting
