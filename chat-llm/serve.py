"""Open-weight LLM backend for the portfolio chatbot, served on Modal.

Runs vLLM's OpenAI-compatible server for Qwen2.5-7B-Instruct on one L4 GPU.
The portfolio's Vercel function (portfolio/api/chat.js) calls it with the
system prompt and tools, so this server only hosts the model.

The app scales to zero after SCALEDOWN_SECONDS without traffic, so you pay
only while visitors are chatting. The first message after an idle period
waits for a cold start (GPU boot + model load).

Deploy:
    modal secret create chat-llm-api-key CHAT_LLM_API_KEY=<random string>
    modal deploy chat-llm/serve.py
"""

import subprocess

import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SERVED_MODEL_NAME = "marvin-chat"
VLLM_PORT = 8000
SCALEDOWN_SECONDS = 5 * 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.1.1",
        "huggingface_hub[hf_transfer]==0.34.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Cache the model weights and vLLM's compiled kernels across cold starts.
hf_cache = modal.Volume.from_name("chat-llm-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("chat-llm-vllm-cache", create_if_missing=True)

app = modal.App("marvin-chat-llm")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("chat-llm-api-key")],
    scaledown_window=SCALEDOWN_SECONDS,
    max_containers=1,
    timeout=10 * 60,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        # Requests must send this key as a Bearer token (read from the secret).
        "--api-key", "$CHAT_LLM_API_KEY",
        # Lets the model call the portfolio's card tools (getProjects, ...).
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        # Chat context stays short; a smaller window leaves more GPU memory free.
        "--max-model-len", "8192",
        # Skips CUDA graph capture: slower generation, but a faster cold start,
        # which matters more for an app that scales to zero.
        "--enforce-eager",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
