
# Miner Setup

**Minimum Requirements**
- CPU Virtual Machine
- OpenAI API Key at https://platform.openai.com/
- Anthropic API Key at https://api.anthropic.com/
- Calculate your credit commitment based on model_configs at `cortext/global_config.py`


### Supported Models

The miner supports the following models:

1. OpenAI Models:
   - gpt-4o
   - dall-e-3

2. Anthropic Models:
   - claude-3-5-sonnet-20241022

### Rate Limiting

The miner implements rate limiting based on stake and network parameters:
- Minimum stake requirement: 10,000
- Rate limits are proportionally allocated based on stake
- Credits refresh every 60 seconds
- Each request consumes an amount of credit based on the model config (e.g. dall-e-3 & gpt-4o-8k-tokens is 1 credit per request)
- Minimum credit allocation: 64
- Maximum credit allocation: 256

## Step-by-step setup

1. Install repository:
```bash
git clone https://github.com/corcel-api/cortex.t
cd cortex.t
pip install uv
uv venv
. .venv/bin/activate
uv sync
```

2. Spin up necessary services:

| Service | Default Port | Environment Variable | Process Command |
|---------|-------------|---------------------|-----------------|
| Redis | 6379 | `REDIS_PORT` & `REDIS__PORT` | `. scripts/install_redis.sh` |

3. Run main mining process:
```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
pm2 start python --name "mining" -- -m neurons.miner --netuid 18 --wallet.hotkey default --wallet.name default --axon.port "your-public-port" --miner.credit "your-credit-commitment"
```
Notes:
- `your-public-port` is the port you want to expose your miner on
- `your-credit-commitment` is the amount of credit you want to commit to the network. The more credit you commit, the faster & better rewards you will receive.

