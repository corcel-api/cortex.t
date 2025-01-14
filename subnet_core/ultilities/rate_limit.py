import pandas as pd
from ..global_config import CONFIG


def get_rate_limit_proportion(metagraph, uid: int) -> int:
    stake_array = metagraph.S
    w_uids = [
        i
        for i in range(len(stake_array))
        if stake_array[i] > CONFIG.bandwidth.min_stake
    ]
    if uid not in w_uids:
        return 0
    w_stakes = [stake_array[i] for i in w_uids]
    total_stake = sum(w_stakes)
    normalized_stakes = [stake / total_stake for stake in w_stakes]
    return normalized_stakes[w_uids.index(uid)]
