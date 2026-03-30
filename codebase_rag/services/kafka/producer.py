from __future__ import annotations

import contextlib
import json
from typing import Any

from loguru import logger
from pydantic import BaseModel

from ...config import settings


class KafkaService:
    """Singleton async wrapper around AIOKafkaProducer (repomind-style)."""

    _instance: KafkaService | None = None
    _producer: Any = None
    _start_attempted: bool = False
    _last_start_error: str | None = None

    def __new__(cls) -> KafkaService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def start(self) -> None:
        if self._producer is not None:
            return
        hosts = settings.kafka_bootstrap_servers_list()
        if not hosts:
            logger.warning("Kafka producer disabled: KAFKA_BOOTSTRAP_SERVERS not set")
            self._start_attempted = True
            self._last_start_error = "bootstrap_servers_not_configured"
            return

        from aiokafka import AIOKafkaProducer

        self._start_attempted = True
        producer = AIOKafkaProducer(
            bootstrap_servers=hosts,
            value_serializer=self._serializer,
            acks="all",
        )
        try:
            await producer.start()
        except Exception as exc:
            logger.warning("Failed to start Kafka producer: {}", exc)
            self._last_start_error = str(exc)
            with contextlib.suppress(Exception):
                await producer.stop()
            self._producer = None
            return

        self._producer = producer
        self._last_start_error = None
        logger.info("Kafka producer started bootstrap={}", hosts)

    async def stop(self) -> None:
        if self._producer is None:
            return
        try:
            await self._producer.stop()
        except Exception as exc:
            logger.warning("Error stopping Kafka producer: {}", exc)
        finally:
            self._producer = None
        self._last_start_error = None

    async def send(
        self,
        topic: str,
        value: Any,
        *,
        key: str | bytes | None = None,
    ) -> None:
        """Send one message; log and drop if producer is unavailable."""
        if self._producer is None:
            logger.debug(
                "Dropping Kafka message (producer unavailable) last_error={} topic={}",
                self._last_start_error,
                topic,
            )
            return
        kafka_key: bytes | None = None
        if key is not None:
            kafka_key = key if isinstance(key, bytes) else str(key).encode("utf-8")
        try:
            await self._producer.send(topic, value=value, key=kafka_key)
        except Exception as exc:
            logger.warning("Failed to send Kafka topic={}: {}", topic, exc)

    @staticmethod
    def _serializer(value: Any) -> bytes:
        if isinstance(value, BaseModel):
            to_dump = value.model_dump(mode="json")
        elif isinstance(value, dict):
            to_dump = value
        else:
            to_dump = value
        return json.dumps(to_dump, default=str, separators=(",", ":")).encode("utf-8")


kafka_service = KafkaService()
