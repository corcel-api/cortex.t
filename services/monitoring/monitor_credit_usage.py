from blessed import Terminal
import redis
import time
from subnet_core import CONFIG
import math


def get_bar(percentage, width=50):
    filled = int(width * percentage)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return bar


def format_percentage(percentage):
    return f"{percentage * 100:6.2f}%"


def monitor_serving_counters():
    term = Terminal()
    redis_client = redis.Redis(
        host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db
    )

    with term.fullscreen(), term.hidden_cursor():
        while True:
            # Get all keys matching the pattern
            pattern = f"{CONFIG.redis.miner_manager_key}:*"
            all_keys = redis_client.keys(pattern)

            # Filter and process keys
            counter_data = []
            for key in all_keys:
                key = key.decode("utf-8")
                if ":quota" in key:
                    continue

                uid = key.split(":")[1]
                quota_key = f"{CONFIG.redis.miner_manager_key}:{uid}:quota"

                count = redis_client.get(key)
                quota = redis_client.get(quota_key)

                if count is not None and quota is not None:
                    count = int(count)
                    quota = int(quota)
                    percentage = count / quota if quota > 0 else 0
                    counter_data.append((uid, count, quota, percentage))

            # Sort by percentage
            counter_data.sort(key=lambda x: x[3], reverse=True)

            # Clear screen and print header
            print(term.home + term.clear)
            print(term.bold("Serving Counter Monitor") + term.normal)
            print("=" * term.width)
            print(f"{'UID':>5} {'Count':>10} {'Quota':>10} {'Usage':>8} {'Bar':<50}")
            print("-" * term.width)

            # Print bars
            for uid, count, quota, percentage in counter_data:
                bar = get_bar(percentage)
                color = ""
                if percentage >= 0.8:
                    color = term.red
                elif percentage >= 0.5:
                    color = term.yellow
                else:
                    color = term.green

                print(
                    f"{uid:>5} {count:>10} {quota:>10} "
                    f"{format_percentage(percentage)} "
                    f"{color}{bar}{term.normal}"
                )

            # Print footer
            print("=" * term.width)
            print(f"Total miners: {len(counter_data)}")
            print(f"Press Ctrl+C to exit")

            time.sleep(1)


if __name__ == "__main__":
    try:
        monitor_serving_counters()
    except KeyboardInterrupt:
        print("\nExiting...")
