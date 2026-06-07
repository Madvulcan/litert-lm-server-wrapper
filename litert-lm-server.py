#!/usr/bin/env python3
"""
LiteRT-LM OpenAI-Compatible API Server
Supports Gemma 4 E2B and E4B models with text, vision, and audio input.
Maintains conversation context across multi-turn chats via session tracking.
"""

import os
import sys
import time
import base64
import logging
import json
import tempfile
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("litert-lm-server")

# Global state
engine = None
model_path = None
model_size = "2b"

# Persistent conversations: session_id -> {"conv": Conversation, "messages": [...], "last_access": timestamp}
_conversations = {}
MAX_CONVERSATIONS = 100
CONVERSATION_TIMEOUT_SECS = 600  # 10 min inactivity cleanup

# Model registry
MODELS = {
    "2b": {
        "filename": "gemma-4-E2B-it.litertlm",
        "display_name": "Gemma 4 E2B (LiteRT-LM)",
        "params": "2B",
        "size_gb": 2.4,
    },
    "4b": {
        "filename": "gemma-4-E4B-it.litertlm",
        "display_name": "Gemma 4 E4B (LiteRT-LM)",
        "params": "4B",
        "size_gb": 3.5,
    },
}


def find_model_file(size_key):
    """Find model file in HF cache or local models dir."""
    info = MODELS[size_key]
    filename = info["filename"]
    # Check local models dir first
    local = Path.home() / f"llama-cpp-server/models/{filename}"
    if local.exists():
        resolved = local.resolve()
        if resolved.exists() and resolved.stat().st_size > 1_000_000_000:
            return str(local)
    # Check HF cache
    cache_base = Path.home() / ".cache/huggingface/hub"
    for model_dir in cache_base.glob("models--litert-community--gemma-4-E*-it-litert-lm"):
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            for snap in snapshots.iterdir():
                f = snap / filename
                if f.exists():
                    resolved = f.resolve()
                    if resolved.exists() and resolved.stat().st_size > 1_000_000_000:
                        return str(resolved)
    return None


def load_model(size="2b", path=None):
    global engine, model_path, model_size
    import litert_lm
    model_size = size.lower()
    info = MODELS.get(model_size, MODELS["2b"])
    if path:
        model_path = path
    else:
        model_path = find_model_file(model_size)
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found for size '{size}': {model_path}")
    logger.info(f"Loading {info['display_name']}: {model_path} ({os.path.getsize(model_path)/1024/1024/1024:.1f} GB)")
    engine = litert_lm.Engine(
        model_path=model_path,
        backend=litert_lm.Backend.GPU(),
        vision_backend=litert_lm.Backend.GPU(),
        audio_backend=litert_lm.Backend.CPU(),
        max_num_tokens=4096,
        enable_speculative_decoding=None,
    )
    logger.info(f"Model loaded on GPU ({info['params']})")


def _ensure_model_loaded(model_id):
    """Ensure the requested model is loaded. Swap if different from current."""
    global engine, model_path, model_size

    if not model_id:
        return

    # Extract size from model ID (e.g., "gemma-4-e4b" -> "4b")
    size_key = None
    for key in MODELS:
        if f"gemma-4-e{key}" == model_id or model_id.startswith(f"gemma-4-e{key}"):
            size_key = key
            break

    if not size_key:
        # Unknown model, keep current
        return

    if size_key == model_size and engine is not None:
        # Already loaded
        return

    # Need to swap models
    logger.info(f"Swapping model: {model_size} -> {size_key}")

    # Close all active conversations
    for sid, conv_data in list(_conversations.items()):
        try:
            conv_data["conv"].close()
        except:
            pass
    _conversations.clear()

    # Close old engine
    if engine:
        try:
            engine.close()
        except:
            pass
        engine = None

    # Load new model
    load_model(size=size_key)
    logger.info(f"Model swap complete: {MODELS[size_key]['display_name']}")


def unload_model():
    global engine, model_path, model_size, _conversations
    if engine:
        engine.close()
        engine = None
    model_path = None
    # Close all active conversations
    for sid, conv_data in _conversations.items():
        try:
            conv_data["conv"].close()
        except:
            pass
    _conversations.clear()


def _cleanup_stale_conversations():
    """Remove conversations that have been idle too long."""
    now = time.time()
    stale = [sid for sid, d in _conversations.items() if now - d["last_access"] > CONVERSATION_TIMEOUT_SECS]
    for sid in stale:
        try:
            _conversations[sid]["conv"].close()
        except:
            pass
        del _conversations[sid]
        logger.info(f"Cleaned up stale conversation: {sid[:8]}...")


def _get_or_create_conversation(session_id, temperature=0.1):
    """Get existing conversation or create a new one.
    
    IMPORTANT: LiteRT-LM only supports ONE conversation at a time per engine.
    If a different session_id is requested, we must close the old conversation first.
    """
    import litert_lm

    if session_id in _conversations:
        conv_data = _conversations[session_id]
        conv_data["last_access"] = time.time()
        return conv_data

    # Close any existing conversation for a different session
    # (LiteRT-LM only supports one session at a time)
    for existing_sid, existing_data in list(_conversations.items()):
        if existing_sid != session_id:
            try:
                existing_data["conv"].close()
            except:
                pass
            del _conversations[existing_sid]
            logger.info(f"Closed existing conversation {existing_sid[:8]}... to make room for {session_id[:8]}...")

    # Create new conversation
    conv = engine.create_conversation(
        sampler_config=litert_lm.SamplerConfig(temperature=temperature),
    )
    _conversations[session_id] = {
        "conv": conv,
        "messages": [],
        "last_access": time.time(),
    }
    logger.info(f"New conversation: {session_id[:8]}... (total: {len(_conversations)})")
    return _conversations[session_id]


def _process_content(content):
    """Convert REST API content parts to (text, attachments_list)."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    text_parts = []
    attachments = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype == "text":
            text_parts.append(part.get("text", ""))
        elif ptype == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                header, b64 = url.split(",", 1)
                img_data = base64.b64decode(b64)
                ext = ".png"
                if "jpeg" in header or "jpg" in header:
                    ext = ".jpg"
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(img_data)
                tmp.close()
                attachments.append({"type": "image", "path": tmp.name})
            elif os.path.isfile(url):
                attachments.append({"type": "image", "path": os.path.abspath(url)})
            elif url.startswith("file://") and os.path.isfile(url[7:]):
                attachments.append({"type": "image", "path": os.path.abspath(url[7:])})
        elif ptype == "image":
            img_b64 = part.get("image", "")
            if img_b64:
                img_data = base64.b64decode(img_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_data)
                tmp.close()
                attachments.append({"type": "image", "path": tmp.name})
        elif ptype == "audio_url":
            url = part.get("audio_url", {}).get("url", "")
            if url.startswith("data:"):
                header, b64 = url.split(",", 1)
                audio_data = base64.b64decode(b64)
                ext = ".wav"
                if "mp3" in header:
                    ext = ".mp3"
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(audio_data)
                tmp.close()
                attachments.append({"type": "audio", "path": tmp.name})
            elif os.path.isfile(url):
                attachments.append({"type": "audio", "path": os.path.abspath(url)})
            elif url.startswith("file://") and os.path.isfile(url[7:]):
                attachments.append({"type": "audio", "path": os.path.abspath(url[7:])})
        elif ptype == "audio":
            audio_b64 = part.get("audio", "")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_data)
                tmp.close()
                attachments.append({"type": "audio", "path": tmp.name})
            elif part.get("path") and os.path.isfile(part["path"]):
                attachments.append({"type": "audio", "path": os.path.abspath(part["path"])})

    full_text = " ".join(text_parts) if text_parts else "Describe this"
    return full_text, attachments


def generate(messages, session_id=None, max_tokens=512, temperature=0.1):
    """Generate a response with conversation context.

    Args:
        messages: Full list of messages in the conversation so far.
        session_id: Unique session identifier for conversation persistence.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The generated text response.
    """
    import litert_lm

    if engine is None:
        raise RuntimeError("Model not loaded")

    if not session_id:
        session_id = str(uuid.uuid4())

    conv_data = _get_or_create_conversation(session_id, temperature)
    conv = conv_data["conv"]
    existing_messages = conv_data["messages"]

    tmp_files = []
    try:
        # Determine which messages are new (not yet in the conversation)
        # We compare bycontent to avoid duplicates
        new_messages = []
        if len(messages) > len(existing_messages):
            new_messages = messages[len(existing_messages):]
        elif messages != existing_messages:
            # Full reset — close old, create new
            try:
                conv.close()
            except:
                pass
            conv = engine.create_conversation(
                sampler_config=litert_lm.SamplerConfig(temperature=temperature),
            )
            conv_data["conv"] = conv
            new_messages = messages
            existing_messages = []

        # Send each new message to build context
        response_text = ""
        for msg in new_messages:
            content = msg.get("content", "")
            full_text_output = ""

            if isinstance(content, str):
                stream = conv.send_message_async(content)
            elif isinstance(content, list):
                full_text, attachments = _process_content(content)
                tmp_files.extend(a["path"] for a in attachments)
                if attachments:
                    content_parts = []
                    for att in attachments:
                        if att["type"] == "image":
                            content_parts.append(litert_lm.Content.ImageFile(att["path"]))
                        elif att["type"] == "audio":
                            content_parts.append(litert_lm.Content.AudioFile(att["path"]))
                    content_parts.append(litert_lm.Content.Text(full_text))
                    stream = conv.send_message_async(litert_lm.Contents(content_parts))
                else:
                    stream = conv.send_message_async(full_text)
            else:
                stream = conv.send_message_async(str(content))

            for chunk in stream:
                for item in chunk.get("content", []):
                    if isinstance(item, dict) and item.get("type") == "text":
                        full_text_output += item.get("text", "")

            response_text = full_text_output

        # Update stored messages
        conv_data["messages"] = list(messages)

        return response_text
    finally:
        for tf in tmp_files:
            try:
                os.unlink(tf)
            except:
                pass


def create_app():
    try:
        from fastapi import FastAPI, HTTPException, Header
        from pydantic import BaseModel
        from typing import Optional
    except ImportError:
        os.system(f"{sys.executable} -m pip install fastapi uvicorn")
        from fastapi import FastAPI, HTTPException, Header
        from pydantic import BaseModel
        from typing import Optional

    from fastapi.responses import StreamingResponse

    app = FastAPI(title="LiteRT-LM Server")

    class ChatMessage(BaseModel):
        role: str
        content: str | list[dict]

    class ChatRequest(BaseModel):
        model: str = "gemma-4-e2b"
        messages: list[ChatMessage]
        max_tokens: int = 512
        temperature: float = 0.1
        stream: bool = False
        extra_body: dict = {}

    @app.get("/health")
    async def health():
        info = MODELS.get(model_size, MODELS["2b"])
        return {"status": "ok", "model": f"gemma-4-e{model_size}", "display_name": info["display_name"], "active_conversations": len(_conversations)}

    @app.get("/")
    async def root():
        return await list_models()

    @app.post("/reset")
    async def reset_conversation(x_session_id: Optional[str] = Header(None)):
        """Reset a conversation by session ID."""
        sid = x_session_id or "default"
        if sid in _conversations:
            try:
                _conversations[sid]["conv"].close()
            except:
                pass
            del _conversations[sid]
            return {"status": "reset", "session_id": sid}
        return {"status": "not_found", "session_id": sid}

    @app.get("/v1/models")
    async def list_models():
        data = []
        for key, info in MODELS.items():
            data.append({
                "id": f"gemma-4-e{key}",
                "object": "model",
                "created": 1780257781,
                "owned_by": "litert-lm",
                "root": f"gemma-4-e{key}",
                "parent": None,
                "context_length": 4096,
                "capabilities": ["text", "vision", "audio", "tools"],
                "display_name": info["display_name"],
            })
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest, x_session_id: Optional[str] = Header(None)):
        if engine is None:
            raise HTTPException(503, "Model not loaded")
        try:
            # Auto-switch model based on request
            _ensure_model_loaded(request.model)

            sid = x_session_id or "default"
            messages = [{"role": m.role, "content": m.content if isinstance(m.content, str) else [dict(c) for c in m.content]} for m in request.messages]

            has_image = any(
                isinstance(m.get("content"), list) and any(p.get("type") in ("image_url", "image") for p in m["content"] if isinstance(p, dict))
                for m in messages
            )
            has_audio = any(
                isinstance(m.get("content"), list) and any(p.get("type") in ("audio_url", "audio") for p in m["content"] if isinstance(p, dict))
                for m in messages
            )
            logger.info(f"chat req [{sid[:8]}]: {len(messages)} msgs, max_tokens={request.max_tokens}, stream={request.stream}, image={has_image}, audio={has_audio}")

            import asyncio
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, lambda: generate(messages, session_id=sid, max_tokens=request.max_tokens, temperature=request.temperature))

            pt = sum(len(m.get("content", "").split()) * 2 if isinstance(m.get("content"), str) else 100 for m in messages)
            ct = len(text.split()) * 2

            if request.stream:
                async def stream_gen():
                    chunk = {"id": f"chatcmpl-{int(time.time()*1000)}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    final = {"id": f"chatcmpl-{int(time.time()*1000)}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}, "done": True}
                    yield f"data: {json.dumps(final)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(stream_gen(), media_type="text/event-stream")
            else:
                resp = {"id": f"chatcmpl-{int(time.time()*1000)}", "object": "chat.completion", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}, "done": False}
                return resp
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise HTTPException(500, str(e))

    @app.post("/chat/completions")
    async def chat_completions_alias(request: ChatRequest, x_session_id: Optional[str] = Header(None)):
        return await chat_completions(request, x_session_id)

    @app.get("/api/tags")
    async def ollama_tags():
        models = []
        for key, info in MODELS.items():
            models.append({"name": f"gemma-4-e{key}", "model": f"gemma-4-e{key}", "size": int(info["size_gb"] * 1_000_000_000), "digest": "litert-lm", "details": {"parent_model": "", "format": "litertlm", "family": "gemma4", "families": ["gemma4"], "parameter_size": info["params"]}})
        return {"models": models}

    @app.post("/api/chat")
    async def ollama_chat(request: ChatRequest, x_session_id: Optional[str] = Header(None)):
        return await chat_completions(request, x_session_id)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "server", "benchmark"], default="cli")
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--port", type=int, default=11456)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--model-size", choices=["2b", "4b"], default="2b")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    load_model(size=args.model_size, path=args.model_path)

    if args.mode == "cli":
        content = args.prompt or "Hello"
        if args.image:
            content = [{"type": "image_url", "image_url": {"url": f"file://{args.image}"}}, {"type": "text", "text": content}]
        if args.audio:
            content = [{"type": "audio_url", "audio_url": {"url": f"file://{args.audio}"}}, {"type": "text", "text": content}]
        messages = [{"role": "user", "content": content}]
        t0 = time.time()
        response = generate(messages)
        print(f"\nResponse ({time.time()-t0:.2f}s):\n{response}")
    elif args.mode == "server":
        import uvicorn
        app = create_app()
        logger.info(f"Starting server on 0.0.0.0:{args.port} (model: gemma-4-e{model_size})")
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
