"""Open-weight LLM backend for the portfolio chatbot, served on Modal.

Runs vLLM's OpenAI-compatible server for Qwen2.5-3B-Instruct on one L4 GPU.
The portfolio's Vercel function (portfolio/api/chat.js) calls it with the
system prompt and tools, so this server only hosts the model.

The app scales to zero after SCALEDOWN_SECONDS without traffic, so you pay
only while visitors are chatting. To keep cold starts short it uses Modal's
GPU memory snapshots: the first container boots vLLM, warms it up, puts it to
sleep (weights moved off the GPU) and is snapshotted; later containers restore
that snapshot and only wake vLLM, instead of loading the model from scratch.

Every request must carry a Modal proxy auth token (Modal-Key / Modal-Secret
headers). vLLM's sleep/wake endpoints are only exposed in dev mode and its own
--api-key doesn't cover them, so the proxy auth is what keeps them private.

Deploy:
    modal deploy chat-llm/serve.py
"""

import json
import subprocess
import time
import urllib.error
import urllib.request

import modal

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SERVED_MODEL_NAME = "marvin-chat"
VLLM_PORT = 8000
SCALEDOWN_SECONDS = 5 * 60
LOCAL_URL = f"http://127.0.0.1:{VLLM_PORT}"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.1.1",
        "huggingface_hub[hf_transfer]==0.34.4",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Exposes /sleep and /wake_up, which the snapshot lifecycle below uses.
        "VLLM_SERVER_DEV_MODE": "1",
    })
)

# Cache the model weights and vLLM's compiled kernels across cold starts.
hf_cache = modal.Volume.from_name("chat-llm-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("chat-llm-vllm-cache", create_if_missing=True)

app = modal.App("marvin-chat-llm")


def _request(method, path, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        LOCAL_URL + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _wait_until(check, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM did not {what} within {timeout} s")


def _is_sleeping():
    return json.loads(_request("GET", "/is_sleeping"))["is_sleeping"]


@app.cls(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    scaledown_window=SCALEDOWN_SECONDS,
    max_containers=1,
    timeout=10 * 60,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=16)
class Server:
    @modal.enter(snap=True)
    def start_and_sleep(self):
        """Runs once, before the snapshot is taken."""
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--served-model-name", SERVED_MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            # Lets the model call the portfolio's card tools (getProjects, ...).
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            # Chat context stays short; a smaller window leaves more GPU memory free.
            "--max-model-len", "8192",
            # Lets vLLM release GPU memory before the snapshot (see below).
            "--enable-sleep-mode",
            # Skips CUDA graph capture: slightly slower generation, fewer moving
            # parts for the snapshot/restore cycle.
            "--enforce-eager",
        ]
        self.process = subprocess.Popen(cmd)
        _wait_until(lambda: _request("GET", "/health") is not None, 10 * 60, "start")

        # One real request so lazy initialisation happens before the snapshot.
        _request("POST", "/v1/chat/completions", {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }, timeout=120)

        # Level 1 moves the weights to CPU memory and drops the KV cache, so the
        # GPU part of the snapshot is small and fast to restore.
        _request("POST", "/sleep?level=1", timeout=120)
        _wait_until(_is_sleeping, 120, "go to sleep")

    @modal.enter(snap=False)
    def wake(self):
        """Runs in every container restored from the snapshot."""
        _request("POST", "/wake_up", timeout=120)
        _wait_until(lambda: not _is_sleeping(), 120, "wake up")

    # The label keeps the URL the same as the earlier function-based version:
    # https://<workspace>--marvin-chat-llm-serve.modal.run
    @modal.web_server(
        port=VLLM_PORT,
        startup_timeout=10 * 60,
        requires_proxy_auth=True,
        label="marvin-chat-llm-serve",
    )
    def serve(self):
        # vLLM is already listening; it was started in start_and_sleep.
        pass
