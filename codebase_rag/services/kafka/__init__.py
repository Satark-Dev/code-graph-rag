from __future__ import annotations

from typing import Any

from .index_job_payload import IndexJobPayload

__all__ = [
    "IndexJobPayload",
    "enqueue_chat_job",
    "enqueue_index_job",
    "ensure_index_job_consumer_started",
    "get_chat_job_key",
    "get_chat_jobs_topic",
    "get_index_job_key",
    "get_index_jobs_topic",
    "kafka_service",
    "run_index_job_consumer",
    "start_index_job_consumer_background",
    "stop_index_job_consumer_background",
]


def __getattr__(name: str) -> Any:
    if name == "kafka_service":
        from .producer import kafka_service as ks

        return ks
    if name == "enqueue_chat_job":
        from .chat_jobs import enqueue_chat_job as fn

        return fn
    if name == "get_chat_jobs_topic":
        from .chat_jobs import get_chat_jobs_topic as fn

        return fn
    if name == "get_chat_job_key":
        from .chat_jobs import get_chat_job_key as fn

        return fn
    if name == "enqueue_index_job":
        from .index_jobs import enqueue_index_job as fn

        return fn
    if name == "get_index_jobs_topic":
        from .index_jobs import get_index_jobs_topic as fn

        return fn
    if name == "get_index_job_key":
        from .index_jobs import get_index_job_key as fn

        return fn
    if name == "run_index_job_consumer":
        from .index_job_consumer import run_index_job_consumer

        return run_index_job_consumer
    if name in (
        "ensure_index_job_consumer_started",
        "start_index_job_consumer_background",
        "stop_index_job_consumer_background",
    ):
        from . import index_job_consumer_controller as c

        return getattr(c, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
