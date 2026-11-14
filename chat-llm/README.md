# Portfolio chatbot LLM (Modal)

Serves the open-weight model behind the portfolio's "Chat with me" page.
`serve.py` runs [vLLM](https://docs.vllm.ai)'s OpenAI-compatible server for
**Qwen2.5-7B-Instruct** on one **L4 GPU** on [Modal](https://modal.com).
The Vercel function `portfolio/api/chat.js` sends it the system prompt and
the card tools (`getProjects`, `getSkills`, ...) and streams the reply to the
site.

The app scales to zero after 5 minutes without traffic, so it costs nothing
while idle. The first message after an idle period waits for a cold start
(GPU boot plus model load); the chat page shows a "waking up" note meanwhile.

## Deploy

1. Create the shared key. Any long random string works, for example from
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`:

   ```
   modal secret create chat-llm-api-key CHAT_LLM_API_KEY=<the key>
   ```

2. Deploy:

   ```
   modal deploy chat-llm/serve.py
   ```

   Modal prints the endpoint URL, like
   `https://<workspace>--marvin-chat-llm-serve.modal.run`.

3. In Vercel → the portfolio project → Settings → Environment Variables, add
   for both Production and Preview:

   | Name | Value |
   | --- | --- |
   | `CHAT_LLM_BASE_URL` | the URL from step 2, plus `/v1` |
   | `CHAT_LLM_API_KEY` | the key from step 1 |

   Then redeploy the project so the function picks them up.

The first deploy downloads the model weights (about 15 GB) into the
`chat-llm-hf-cache` volume; later cold starts load them from there.

## Cost

Billing is per second of GPU time while a container runs. Each visit costs
the cold start, the chat itself, and the 5 idle minutes before scale-down.
To cap spending, set a workspace budget in Modal's billing settings.
