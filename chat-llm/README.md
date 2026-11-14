# Portfolio chatbot LLM (Modal)

Serves the open-weight model behind the portfolio's "Chat with me" page.
`serve.py` runs [vLLM](https://docs.vllm.ai)'s OpenAI-compatible server for
**Qwen2.5-3B-Instruct** on one **L4 GPU** on [Modal](https://modal.com).
The Vercel function `portfolio/api/chat.js` sends it the system prompt and
the card tools (`getProjects`, `getSkills`, ...) and streams the reply to the
site.

## Keeping cold starts short

The app scales to zero after 5 minutes without traffic, so it costs nothing
while idle. Three things keep the first reply after an idle period quick:

- **GPU memory snapshot.** The first container after a deploy boots vLLM,
  sends one warm-up request, puts vLLM to sleep (weights moved off the GPU)
  and is snapshotted. Later containers restore that snapshot and only wake
  vLLM, instead of loading the model from scratch. The first cold start after
  each deploy is the slow one, because it creates the snapshot.
- **Small model.** A 3B model has less to restore and wake than a 7B one.
- **Warm-up on page open.** The chat page calls `GET /api/chat` when it opens,
  which pings the model's `/health`, so it boots while the visitor types.

GPU memory snapshots are an experimental Modal feature. If a restore fails,
Modal falls back to a normal (slow) cold start.

## Security

Every request needs a Modal proxy auth token (`Modal-Key` / `Modal-Secret`
headers), checked by Modal before it reaches vLLM. This matters because the
snapshot lifecycle needs vLLM's dev-mode `/sleep` and `/wake_up` endpoints,
which vLLM's own `--api-key` does not protect.

## Deploy

1. Create a proxy auth token: Modal dashboard → workspace **Settings** →
   **Proxy Auth Tokens** → create one. Copy its token ID (`wk-...`) and
   secret (`ws-...`); the secret is shown only once.

2. Deploy:

   ```
   modal deploy chat-llm/serve.py
   ```

   Modal prints the endpoint URL,
   `https://<workspace>--marvin-chat-llm-serve.modal.run`.

3. In Vercel → the portfolio project → Settings → Environments, set for both
   Production and Preview:

   | Name | Value |
   | --- | --- |
   | `CHAT_LLM_BASE_URL` | the URL from step 2, plus `/v1` |
   | `CHAT_LLM_MODAL_KEY` | the token ID (`wk-...`) |
   | `CHAT_LLM_MODAL_SECRET` | the token secret (`ws-...`) |

   Then redeploy the project so the function picks them up.

## Cost

Billing is per second of GPU time while a container runs. Each visit costs
the cold start, the chat itself, and the 5 idle minutes before scale-down;
visitors who open the chat page without sending anything still trigger a
warm-up. To cap spending, set a workspace budget in Modal's billing settings.
