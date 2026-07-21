# Goose (Block) + Lumen Integration

[Goose](https://github.com/block/goose) is an open-source AI agent from Block (formerly Square) that supports **MCP servers** natively.

## Configuration

Add Lumen to your Goose configuration:

```bash
goose configure
# Select "Add Extension" → "Command-line Extension"
# Name: lumen-memory
# Command: python -m lumen.integrations.mcp_server
# Environment variables:
#   LUMEN_DEVICE=generic
```

Or edit `~/.config/goose/config.yaml` directly:

```yaml
extensions:
  lumen-memory:
    type: stdio
    cmd: python
    args:
      - -m
      - lumen.integrations.mcp_server
    env:
      LUMEN_DEVICE: generic
      LUMEN_STORE_PATH: "${HOME}/.lumen/store"
      LUMEN_MODEL_PATH: "${HOME}/.lumen/models"
```

## Usage

Start Goose with Lumen enabled:

```bash
goose session
```

Inside the session:

```
> Search my memory palace for decisions about caching
> Remember that we use Redis for session storage
> Assemble context for implementing OAuth
> What is the status of my memory palace?
```

Goose will automatically route these to the Lumen MCP tools.

## Advanced: Goose + Lumen in CI/CD

Use Goose with Lumen for persistent CI/CD agents:

```bash
export LUMEN_API_KEY=your-key
goose session --instructions "Always search Lumen before making architectural decisions."
```
