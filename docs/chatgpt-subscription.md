# Running the TUI on a ChatGPT subscription quota (Plus / Pro)

This guide makes the heartbit TUI run on the **usage quota of a ChatGPT
subscription** (Plus / Pro / Team / Enterprise) instead of pay-per-token API
billing.

> ## ⚠️ Read this first — Terms of Service & risk
>
> A ChatGPT subscription is **not** an OpenAI API plan. The only token that bills
> model usage to your subscription is the **OAuth token Codex obtains via "Sign
> in with ChatGPT"** (stored in `~/.codex/auth.json`). OpenAI sanctions that token
> **only inside the Codex product** (CLI / IDE / Cloud).
>
> Using it to drive a third-party agent (this TUI) goes through a **local proxy
> that mimics Codex requests** — this is **unofficial and a likely Terms-of-
> Service violation**, with a real risk of **account suspension/ban**. It is also
> **fragile**: it depends on the Codex endpoint shape and a server-side
> "looks-like-Codex" auth check that OpenAI changes frequently, so it breaks
> without notice.
>
> Use it for **personal, local experimentation on your own machine and your own
> subscription only**. Do **not** host it, share the token, or pool tokens across
> accounts. If you want zero ToS risk, set `HEARTBIT_OPENAI_BASE_URL` to the real
> OpenAI API (`https://api.openai.com/v1`) with an API key, to OpenRouter, or to a
> local model — the same TUI feature supports all of those.

## How it works

```
ChatGPT login  →  ~/.codex/auth.json   (access_token + account_id)
                        │
            local proxy (codex-openai-proxy / openai-oauth)
                        │  translates OpenAI /chat/completions  →  Codex /responses
                        ▼
   http://127.0.0.1:<port>/v1   ← heartbit TUI points here
```

The proxy reads your Codex token, exposes an **OpenAI-compatible** endpoint on
localhost, and rewrites each request into the Codex "Responses" shape (injecting
the Codex system prompt so OpenAI's auth check passes). The heartbit TUI then
talks to that endpoint like any OpenAI-compatible API — and the usage is billed
to your **subscription quota**.

## Setup

### 1. Authenticate Codex with your ChatGPT subscription

```bash
npm install -g @openai/codex      # or: brew install codex
codex login                        # opens a browser → "Sign in with ChatGPT"
```

This writes `~/.codex/auth.json`. Treat that file like a password.

### 2. Run a local Codex → OpenAI-compatible proxy

Pick a maintained proxy (it tracks the changing Codex wire format so you don't
have to):

- [`Securiteru/codex-openai-proxy`](https://github.com/Securiteru/codex-openai-proxy)
  — built for "use a ChatGPT Plus token via OpenAI API compatibility".
- [`EvanZhouDev/openai-oauth`](https://github.com/EvanZhouDev/openai-oauth) —
  localhost proxy pre-authenticated from `~/.codex/auth.json`.

Follow that project's README to start it; note the port (e.g. `10531`). You'll
get an endpoint like `http://127.0.0.1:10531/v1`.

### 3. Point the heartbit TUI at the proxy

#### Option A — one command, inside the TUI (recommended)

Start the TUI, then run:

```
/codex
```

That single command points the engine at the default proxy URL
(`http://127.0.0.1:10531/v1`), switches the model to `gpt-5-codex`, and respawns
the agent — all at once. Pass a URL to override the port
(`/codex http://127.0.0.1:8080/v1`), and `/codex off` reverts to your normal
provider. It prints the ToS caveat and warns if `~/.codex/auth.json` is missing.

`/codex` is a **session override** — it does not write anything to your config, so
the Codex model id can't leak into the next launch and break it once the proxy is
gone. `/codex off` restores the model and provider you had before.

#### Option B — env vars (set before launch)

The TUI also honours a **custom OpenAI-compatible endpoint** via two env vars
(see `build_provider` in `crates/heartbit-tui/src/main.rs`):

| Var | Meaning |
|-----|---------|
| `HEARTBIT_OPENAI_BASE_URL` | the proxy's `/v1` base URL. **Set this.** Takes priority over OpenRouter. |
| `HEARTBIT_OPENAI_API_KEY`  | optional. **Leave UNSET** for a localhost proxy — an empty key uses `AuthStyle::None`, which is what allows a non-HTTPS `http://127.0.0.1` URL. Set it only for an HTTPS endpoint that needs a bearer key. |

```bash
export HEARTBIT_OPENAI_BASE_URL="http://127.0.0.1:10531/v1"
# do NOT set HEARTBIT_OPENAI_API_KEY for a localhost http proxy
cargo run -p heartbit-tui          # or your installed `heartbit` TUI binary
```

On startup the TUI prints a notice confirming the custom endpoint. Then set the
model to one the Codex backend exposes:

```
/model gpt-5-codex
```

(Use whatever model id the proxy advertises — the Codex backend exposes its own
model set, not the full OpenAI catalogue. Check the proxy's `/v1/models`.)

## Notes & troubleshooting

- **Pro vs Plus quota.** Pro gives much higher Codex limits than Plus, but it is
  still a **rate/usage quota**, not unlimited API — heavy agent runs will get
  throttled. There is no official "5×" SKU; Pro is the highest standard tier.
- **Token expiry.** The Codex token expires; re-run `codex login` (and restart
  the proxy) when requests start failing with auth errors.
- **HTTP vs HTTPS.** A localhost proxy is `http://` — keep `HEARTBIT_OPENAI_API_KEY`
  unset so the provider uses `AuthStyle::None` (the only style that permits a
  non-HTTPS base URL). Setting a key forces Bearer + HTTPS and will reject an
  `http://` URL.
- **Model behaviour.** The Codex backend is tuned for coding; tool-calling and
  reasoning behaviour may differ from general OpenRouter models.
- **Zero-risk alternative.** The same `HEARTBIT_OPENAI_BASE_URL` mechanism points
  at the real OpenAI API (`https://api.openai.com/v1` + `HEARTBIT_OPENAI_API_KEY`),
  a local model (Ollama/vLLM/LM Studio), or you can keep using OpenRouter (`/key`).
