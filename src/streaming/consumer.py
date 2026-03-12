import json
import os
import time
import requests
from confluent_kafka import Consumer

BROKER = os.getenv("KAFKA_BROKER", "redpanda:9092")
TOPIC = os.getenv("TX_TOPIC", "transactions")
GROUP = os.getenv("TX_GROUP", "fraud-consumer")
SCORING_URL = os.getenv("SCORING_URL", "http://fraud_api:8000/score")

def main():
    c = Consumer(
        {
            "bootstrap.servers": BROKER,
            "group.id": GROUP,
            "auto.offset.reset": "earliest",
        }
    )
    c.subscribe([TOPIC])
    print("consumer started:", BROKER, TOPIC, "->", SCORING_URL)

    while True:
        msg = c.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print("kafka error:", msg.error())
            continue

        tx = json.loads(msg.value().decode("utf-8"))
        try:
            r = requests.post(SCORING_URL, json=tx, timeout=2)
            r.raise_for_status()
            out = r.json()
            print("scored", out["transaction_id"], "score=", out["score"], "decision=", out["decision"])
        except Exception as e:
            print("score failed:", e)
            time.sleep(1)

if __name__ == "__main__":
    main()
