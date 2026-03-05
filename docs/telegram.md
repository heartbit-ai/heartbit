# Telegram Bot Integration

Heartbit integrates with Telegram as an interactive channel for the daemon mode.

## Features

- **Direct messages** — users interact with the agent via Telegram DMs
- **Streaming responses** — agent responses stream in real-time
- **Human-in-the-loop (HITL)** — approval buttons for tool execution
- **Keyboard menus** — structured question/answer via Telegram keyboard buttons
- **Access control** — whitelist users/groups, rate limiting
- **Session management** — per-chat session context

## Configuration

Set the Telegram bot token via environment variable:

```bash
export HEARTBIT_TELEGRAM_TOKEN=<your-bot-token>
```

Or in the daemon config:

```toml
[daemon]
bind = "127.0.0.1:3000"

# Telegram is activated when the token is present in the environment
# Additional Telegram-specific settings are in the daemon config
```

## Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Set the `HEARTBIT_TELEGRAM_TOKEN` environment variable
3. Start the daemon: `heartbit daemon --config heartbit.toml`
4. Message your bot on Telegram

## Architecture

The Telegram adapter (`TelegramBridge`) implements the `InteractionBridge` trait, connecting Telegram's messaging API to the agent's callback system:

- `OnText` — streams text responses back to Telegram
- `OnInput` — receives user input from Telegram messages
- `OnApproval` — sends approval buttons, waits for user response
- `OnQuestion` — sends structured questions with keyboard buttons

## Feature Flag

Telegram support requires the `telegram` feature flag:

```bash
cargo build --features telegram
```

The CLI binary (`heartbit-cli`) includes it via the `full` feature.
