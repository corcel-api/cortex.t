## Overview

The system operates as follows:

1. **Model and Credit Allocation:** The owner defines the model and sets the credit cost per request.
2. **Miner Commitments:** Miners commit a total amount of credits per interval to validators.
3. **Validator Quotas:** Validators receive a credit quota proportional to their stake, normalized by the total stake.

### Example Configuration
```yaml
- interval: 60 seconds
- model-config:
    - gpt-4o-mini: quota=1
    - gpt-4o: quota=4
- miner-commit:
    - uid-2: 128
    - uid-3: 156
    - uid-4: 184
- validator-stake:
    - uid-0: 10000
    - uid-1: 10000
```

**Resulting Validator Quotas:**
```yaml
- validator-quota:
    - uid-0: [64, 78, 92]
    - uid-1: [64, 78, 92]
```

### Quota Usage
- **Redis Queue:** Validators maintain a Redis queue to track quota usage as counters.
- **Synthetic Process:** Validators run a synthetic process to iteratively consume batches of 4 miners for performance validation. Each batch:
  - Consumes up to 50% of each miner's quota.
  - Randomly selects a model from `model-config`.
- **Organic Server:** A separate organic server consumes quota directly from the Redis queue, with up to 100% quota usage for each miner.

## Setting Up

See [VALIDATOR.md](docs/VALIDATOR.md) for more details.
