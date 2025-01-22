## Validator setup

**Minimum Requirements**
- CPU is only needed for validating.
- OpenAI API key is needed for scoring. https://platform.openai.com/

Validator maintains some services and database for validating and organic serving:
**Database**
- Redis: rate limit counter, some of organic payloads that will be used to re-validate in synthetic validating.
- SQLite: store score and metadata of miners.

**Restfull API**
- Wrap Subtensor: expose API for query bittensor functions: metagraph, set weights, auto-sync.
    - `POST /set_weights`: Do set weights with current score from SQLite.
    - `POST /axons`: Get axons information (ip, port, public key, etc.) from a set of UIDs.
- Synthesizing: generate synthetic payloads for validating.
    - `POST /synthesize`: return miner payload of synapse.
- Scoring: score miner payloads.
    - `POST /score`: score miner payloads.
- Organic: expose endpoint for serving organic payloads.
    - `POST /organic`: serve organic payloads.

### Step-by-step setup

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
export NETUID=18
export WALLET_NAME=default
export WALLET_HOTKEY=default
export AXON_PORT=8000
export OPENAI_API_KEY=your_openai_api_key
```

To print all configuration:
```
python -m cortext
```

3. Spin up necessary services:
```
. scripts/install_redis.sh
pm2 start python --name "cortex_w_subtensor" -- -m services.subtensor_syncing.server
pm2 start python --name "cortex_scoring" -- -m services.scoring.server
pm2 start python --name "cortex_synthesizing" -- -m services.synthesizing.server
pm2 start python --name "cortex_managing" -- -m services.managing.server
pm2 start python --name "cortex_synthesizing_worker" -- -m services.synthesizing.refill_worker
```

4. Run main validating proccess:
```
pm2 start python --name "cortex_validating" -- -m neurons.validator
```
