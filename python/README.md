# Predicato Python Client

Python client for the Predicato knowledge graph framework.

## Installation

```bash
pip install predicato
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from predicato import PredicatoClient

# Initialize the client
client = PredicatoClient(
    base_url="http://localhost:8080",
    group_id="my-project"
)

# Ingest an episode
result = client.add_episode(
    name="Meeting Notes",
    content="Discussed project timeline and milestones...",
    source="meeting"
)

print(f"Created episode: {result.episode}")
print(f"Extracted {len(result.nodes)} entities")

# Search the knowledge graph
results = client.search(query="project timeline", limit=10)
for node in results.nodes:
    print(f"- {node.name} ({node.type})")

# Clean up
client.close()
```

### Using Context Manager

```python
from predicato import PredicatoClient

with PredicatoClient(base_url="http://localhost:8080") as client:
    result = client.add_episode(
        name="Document",
        content="...",
        group_id="my-project"
    )
```

### Async Usage

```python
import asyncio
from predicato import AsyncPredicatoClient

async def main():
    async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
        result = await client.add_episode(
            name="Document",
            content="...",
            group_id="my-project"
        )

asyncio.run(main())
```

## Configuration

The client can be configured via constructor parameters or environment variables:

| Parameter | Environment Variable | Description |
|-----------|---------------------|-------------|
| `base_url` | `PREDICATO_URL` | Server URL |
| `group_id` | `PREDICATO_GROUP_ID` | Default group ID |
| `timeout` | - | Request timeout in seconds (default: 30) |

## API Reference

### Episode Ingestion

```python
# Single episode
result = client.add_episode(
    name="Document Title",
    content="Document content...",
    source="document",           # optional
    group_id="my-group",         # optional if set in constructor
    metadata={"key": "value"}    # optional
)

# Bulk episodes
results = client.add_episodes([
    Episode(name="Doc 1", content="...", group_id="my-group"),
    Episode(name="Doc 2", content="...", group_id="my-group"),
])
```

### Message Ingestion

```python
from predicato import Message

result = client.add_messages(
    messages=[
        Message(role="user", content="What is the project status?"),
        Message(role="assistant", content="The project is on track..."),
    ],
    group_id="conversation-001"
)
```

### Entity Creation

```python
result = client.add_entity(
    name="Alice Smith",
    entity_type="Person",
    attributes={"role": "Engineer", "department": "Platform"},
    group_id="my-group"
)
```

### Search

```python
results = client.search(
    query="project timeline",
    group_id="my-group",      # optional
    limit=10,                  # optional, default 10
    include_edges=True,        # optional, default False
    min_score=0.5              # optional, default 0.0
)
```

### Data Clearing

```python
result = client.clear_graph(group_ids=["my-group"])
```

## License

MIT
