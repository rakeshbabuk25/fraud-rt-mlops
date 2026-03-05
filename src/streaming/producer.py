import json
import os
import random
import time
from datetime import datetime, timezone

from confluent_kafka import Producer

BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("TX_TOPIC", "transactions")

def make_tx(i: int) -> dict:
    amount = round(random.uniform(1, 3000), 2)
    country = random.choice(["GB", "GB", "GB", "US", "FR", "NG"])  # skew to GB
    mcc = random.choice(["retail", "grocery", "travel", "crypto", "electronics"])
    return {
        "transaction_id": f"tx_{i:06d}",
        "event_time": datetime.now(timezone.utc).isoformat(),
        "amount": amount,
        "currency": "GBP",
        "merchant_category": mcc,
        "country": country,
        "features": {
            "device_age_days": random.randint(0, 365),
            "is_new_customer": random.choice([True, False, False]),
        },
    }

def main():
    p = Producer({"bootstrap.servers": BROKER})
    for i in range(1, 51):
        tx = make_tx(i)
        p.produce(TOPIC, json.dumps(tx).encode("utf-8"))
        p.poll(0)
        print("sent", tx["transaction_id"], tx["amount"], tx["merchant_category"], tx["country"])
        time.sleep(0.1)
    p.flush()

if __name__ == "__main__":
    main()
