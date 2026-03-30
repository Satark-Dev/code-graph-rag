import json
import os
import sys
from collections.abc import Iterable

try:
    from kafka import KafkaConsumer
except ImportError:
    print("Error: 'kafka-python' is not installed. Please run 'pip install kafka-python-ng'")
    sys.exit(1)


TOPICS: Iterable[str] = ["queue.ai.invocation.logs"]


def main() -> None:
    """Simple CLI to inspect AI observability events on Kafka.

    Usage:
        python -m scripts.monitor_kafka
    """
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    try:
        consumer = KafkaConsumer(
            *TOPICS,
            bootstrap_servers=bootstrap,
            enable_auto_commit=True,
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="repomind-observability-monitor",
        )
    except Exception as e:
        print(f"Failed to connect to Kafka at {bootstrap}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Consuming topics={list(TOPICS)} from bootstrap_servers={bootstrap}")
    try:
        for msg in consumer:
            print("-" * 80)
            print(f"topic={msg.topic} partition={msg.partition} offset={msg.offset}")
            try:
                print(json.dumps(msg.value, indent=2, sort_keys=True))
            except (TypeError, ValueError):
                # Fallback if message is not json-serializable
                print(msg.value)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
