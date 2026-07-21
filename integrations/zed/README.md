# Zed + Lumen Integration

[Zed](https://zed.dev) is a high-performance AI-native code editor. Zed supports **context servers** (their term for MCP-like integrations) and **slash commands**.

## Option 1: Zed Context Server (Recommended)

Add to your Zed `settings.json` (via `cmd+,` or `ctrl+,`):

```json
{
  "context_servers": {
    "lumen-memory": {
      "command": {
        "path": "python",
        "args": ["-m", "lumen.integrations.mcp_server"]
      },
      "settings": {
        "LUMEN_DEVICE": "generic"
      }
    }
  }
}
```

Restart Zed. The Lumen tools will appear in the AI assistant panel.

## Option 2: Zed Slash Command (Workaround)

If context servers are not available in your Zed version, create a custom slash command:

Add to `~/.config/zed/settings.json`:

```json
{
  "assistant": {
    "default_model": {
      "provider": "openai",
      "model": "gpt-4o"
    }
  }
}
```

Then create a Zed extension or use the inline assistant with a custom prompt that calls Lumen via HTTP. Zed's extension API for custom slash commands is evolving; check the [Zed extensions docs](https://zed.dev/docs/extensions) for the latest.

## Option 3: Zed Extension (Advanced)

Create `~/.config/zed/extensions/lumen/extension.json`:

```json
{
  "id": "lumen-memory",
  "name": "Lumen Memory",
  "version": "0.1.0",
  "authors": ["Lumen Contributors"],
  "description": "Integrate Lumen memory palace with Zed AI assistant",
  "repository": "https://github.com/QuantumindSSI/lumen"
}
```

And `~/.config/zed/extensions/lumen/src/lib.rs` (WASM extension that calls Lumen HTTP API). This requires Rust/WASM knowledge. For a pilot, Option 1 (context server) is sufficient.

## Quick Test

1. Start Lumen: `lumen serve`
2. Open Zed AI panel (`ctrl+~` or `cmd+~`)
3. Type: `@lumen-memory search for decisions about caching`
