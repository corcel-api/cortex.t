# Validator Setup

![Validator Diagram](../assets/validator-diagram.png)

**Minimum Requirements**
- CPU Virtual Machine
- OpenAI API Key at https://platform.openai.com/
- Claude API Key at https://www.anthropic.com/api
- OpenRouter API Key at https://openrouter.ai

## Step-by-step setup

1. Install repository:
```
git clone https://github.com/corcel-api/cortex.t
cd cortex.t
pip install uv
uv venv
. .venv/bin/activate
uv sync
```

2. Configure env variables:
```
export SUBTENSOR_NETWORK=finney
export SUBTENSOR_NETUID=18
export WALLET_NAME=default
export WALLET_HOTKEY=default
export AXON_PORT=8000
export OPENAI_API_KEY=your_openai_api_key
export OPENROUTER_API_KEY=your_openrouter_api_key
export CLAUDE_API_KEY=your_claude_api_key
```

To verify the exported variables:
```
python -m cortext
```

3. Spin up necessary services:

| Service | Default Port | Environment Variable | Process Command |
|---------|-------------|---------------------|-----------------|
| Redis | 6379 | `REDIS_PORT` & `REDIS__PORT` | `. scripts/install_redis.sh` |
| ExifTool | N/A | N/A | `. scripts/install_exiftool.sh` |
| Subtensor Sync | 8104 | `W_SUBTENSOR__PORT` | `pm2 start python --name "cortex_w_subtensor" -- -m services.subtensor_syncing.server` |
| Scoring | 8101 | `SCORE__PORT` | `pm2 start python --name "cortex_scoring" -- -m services.scoring.server` |
| Synthesizing | 8102 | `SYNTHESIZE__PORT` | `pm2 start python --name "cortex_synthesizing" -- -m services.synthesizing.server` |
| Managing | 8103 | `MINER_MANAGER__PORT` | `pm2 start python --name "cortex_managing" -- -m services.managing.server` |
| Synthesizing Worker | N/A | N/A | `pm2 start python --name "cortex_synthesizing_worker" -- -m services.synthesizing.refill_worker` |


You can modify the default ports by setting the corresponding environment variables before starting the services.

4. Run main validating proccess:
```
pm2 start python --name "cortex_validating" -- -m neurons.validator
```

5. Run the auto update script, it will check for updates every 30 minutes
```bash
pm2 start auto-update.sh --name "auto_updater"
```

## Organic Serving [Optional]
1. Set admin api key
```
export ADMIN_API_KEY=your_admin_api_key
```
2. Run organic proxy server

By default, the organic proxy server is set to listen on port `localhost:8105`.
You can modify the port by setting the `ORGANIC__PORT` environment variable.

```
pm2 start python --name "cortex_organic" -- -m services.organic.server
```
3. Optional: Run organic frontend for managing API keys and Test generation
```
pm2 start python --name "cortex_organic_frontend" -- -m services.organic.frontend
```

| Endpoint | Method | Description | Request Parameters | Authentication | Response |
|----------|---------|-------------|-------------------|----------------|-----------|
| `/api/v1/chat/completions` | POST | Process chat completion requests | `request`: MinerPayload object | API Key required | Streaming response of chat completion chunks |
| `/api/v1/keys` | POST | Create new API key | `user_id`: string<br>`initial_credits`: float (default: 100.0)<br>`monthly_reset`: boolean (default: true) | Admin API Key required | New APIKey object |
| `/api/v1/keys` | GET | Retrieve all API keys | None | Admin API Key required | List of APIKey objects |
| `/api/v1/keys/{key}/add-credits` | POST | Add credits to existing API key | `key`: string (path)<br>`amount`: float | Admin API Key required | Updated APIKey object |
| `/api/v1/keys/{key}/status` | PATCH | Update API key status | `key`: string (path)<br>`status_update`: dict with `is_active` boolean | Admin API Key required | Updated APIKey object |
| `/api/v1/keys/{key}` | DELETE | Delete an API key | `key`: string (path) | Admin API Key required | Success message |

**APIKey Object Structure:**
```python
{
    "key": string,
    "user_id": string,
    "created_at": datetime,
    "is_active": boolean,
    "permissions": list[string],
    "total_credits": decimal,
    "used_credits": decimal,
    "credit_reset_date": datetime (optional)
}
```

**Notes:**
- All endpoints require authentication via the `X-API-Key` header
- Admin operations require the admin API key (set via `ADMIN_API_KEY` environment variable)
- The chat completions endpoint consumes credits based on the model configuration
- Credits can be set to reset monthly if specified during key creation


### Default Config
```
{
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'organic_queue_key': 'organic_queue',
        'synthetic_queue_key': 'synthetic_queue',
        'miner_manager_key': 'node_manager'
    },
    'bandwidth': {
        'interval': 60,
        'min_stake': 10000,
        'model_configs': {
            'gpt-4o': {
                'credit': 1,
                'model': 'gpt-4o',
                'max_tokens': 8096,
                'synapse_type': 'streaming-chat',
                'timeout': 12,
                'allowed_params': ['messages', 'temperature', 'max_tokens', 'stream', 'model', 'seed']
            },
            'dall-e-3': {
                'credit': 1,
                'model': 'dall-e-3',
                'max_tokens': 1024,
                'synapse_type': 'streaming-chat',
                'timeout': 32,
                'allowed_params': ['prompt', 'n', 'size', 'response_format', 'user']
            },
            'claude-3-5-sonnet-20241022': {
                'credit': 1,
                'model': 'claude-3-5-sonnet-20241022',
                'max_tokens': 8096,
                'synapse_type': 'streaming-chat',
                'timeout': 12,
                'allowed_params': ['messages', 'temperature', 'max_tokens', 'stream', 'model']
            }
        },
        'min_credit': 128,
        'max_credit': 1024
    },
    'score': {'decay_factor': 0.9, 'host': 'localhost', 'port': 8101},
    'sql': {'url': 'sqlite:///miner_metadata.db'},
    'network': 'mainnet',
    'synthesize': {'host': 'localhost', 'port': 8102, 'synthetic_pool_size': 1024, 'organic_pool_size': 1024},
    'miner_manager': {'port': 8103, 'host': 'localhost'},
    'validating': {'synthetic_threshold': 0.2, 'synthetic_batch_size': 4, 'synthetic_concurrent_batches': 1},
    'w_subtensor': {'host': 'localhost', 'port': 8104},
    'organic': {'host': 'localhost', 'port': 8105},
    'subtensor_network': 'test',
    'subtensor_netuid': 245,
    'wallet_name': 'tnv',
    'wallet_hotkey': '0',
    'subtensor_tempo': 360,
    'axon_port': 8000
}
```