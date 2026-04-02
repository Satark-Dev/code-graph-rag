"""
Resolve org_id → region → org-database DSN using the Core DB (user_org).

Aligned with the org/region routing pattern (Core DB read-only mapping + per-region org DBs).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


@dataclass
class PostgresConfig:
    host: str
    port: int
    user: str
    password: str
    name: str
    sslmode_required: bool = False


def build_postgres_dsn(config: PostgresConfig) -> str:
    """Build a PostgreSQL DSN from a simple config object."""
    user_q = quote_plus(config.user)
    password_q = quote_plus(config.password)
    ssl_suffix = "?sslmode=require" if config.sslmode_required else ""
    return f"postgresql://{user_q}:{password_q}@{config.host}:{config.port}/{config.name}{ssl_suffix}"


class OrgRegionResolver:
    """Resolves org_id → region number → org DB connection string."""

    def __init__(self, max_cache_size: int = 500) -> None:
        self._max_cache_size = max_cache_size
        self._region_cache: OrderedDict[str, int] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _get_core_db_dsn(self) -> str:
        from ..config import settings

        if settings.CORE_DB_HOST:
            cfg = PostgresConfig(
                host=settings.CORE_DB_HOST,
                port=settings.CORE_DB_PORT or 5432,
                user=settings.CORE_DB_USER or "postgres",
                password=settings.CORE_DB_PASSWORD or "",
                name=settings.CORE_DB_NAME or "postgres",
                sslmode_required=str(settings.CORE_DB_SSL or "").lower() == "true",
            )
            return build_postgres_dsn(cfg)

        raise RuntimeError(
            "Core DB config missing. Set CORE_DB_HOST, CORE_DB_PORT, CORE_DB_USER, "
            "CORE_DB_PASSWORD, CORE_DB_NAME."
        )

    @property
    def core_db_dsn(self) -> str:
        return self._get_core_db_dsn()

    def _org_field(self, region: int, suffix: str) -> str | None:
        from ..config import settings

        key = f"ORG_DB_{suffix}_{region}"
        val = getattr(settings, key, None)
        if val is not None and str(val).strip():
            return str(val)
        env_val = os.getenv(key)
        if env_val is not None and str(env_val).strip():
            return env_val
        return None

    def get_org_db_connection_string(self, region: int) -> str:
        host = self._org_field(region, "HOST")
        if not host:
            raise RuntimeError(
                f"Org DB region {region} is not configured. "
                f"Set ORG_DB_HOST_{region} and related ORG_DB_*_{region} values. "
                "Core DB is only used for org→region lookup, "
                "not as an application data store."
            )

        port_s = self._org_field(region, "PORT") or "5432"
        try:
            port = int(port_s)
        except (TypeError, ValueError):
            port = 5432
        user = self._org_field(region, "USER") or "postgres"
        password = self._org_field(region, "PASSWORD") or ""
        name = self._org_field(region, "NAME") or f"satark_org_db_{region}"
        ssl_val = self._org_field(region, "SSL") or "false"
        cfg = PostgresConfig(
            host=host,
            port=port,
            user=user,
            password=password,
            name=name,
            sslmode_required=str(ssl_val).lower() == "true",
        )
        return build_postgres_dsn(cfg)

    def get_region_for_org_id(self, org_id: str) -> int:
        if not org_id or not str(org_id).strip():
            raise ValueError("org_id cannot be empty.")

        org_id = str(org_id).strip()

        with self._cache_lock:
            if org_id in self._region_cache:
                self._region_cache.move_to_end(org_id)
                return self._region_cache[org_id]

        import psycopg

        from ..config import settings

        logger.debug("Cache miss for org_id=%s — querying Core DB", org_id)
        try:
            conn = psycopg.connect(
                self._get_core_db_dsn(),
                connect_timeout=float(settings.DB_CONNECT_TIMEOUT),
            )
        except Exception:
            logger.exception("Failed to connect to Core DB for org_id=%s", org_id)
            raise

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT region FROM user_org WHERE org_id = %s LIMIT 1",
                    (org_id,),
                )
                row = cur.fetchone()
            if row is None:
                logger.warning(
                    "org_id %s not in user_org. Defaulting to region 1.", org_id
                )
                region = 1
            else:
                region = int(row[0])
        finally:
            conn.close()

        with self._cache_lock:
            self._region_cache[org_id] = region
            if len(self._region_cache) > self._max_cache_size:
                self._region_cache.popitem(last=False)

        return region

    def warm_core_connection(self) -> None:
        import psycopg

        from ..config import settings

        try:
            conn = psycopg.connect(
                self._get_core_db_dsn(),
                connect_timeout=float(settings.DB_CONNECT_TIMEOUT),
            )
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            finally:
                conn.close()
            logger.info("Core DB connection warmed.")
        except Exception as e:
            logger.warning("Failed to warm Core DB: %s", e)


_resolver_lock = threading.Lock()
_resolver: OrgRegionResolver | None = None


def get_org_region_resolver() -> OrgRegionResolver:
    global _resolver
    with _resolver_lock:
        if _resolver is None:
            _resolver = OrgRegionResolver()
        return _resolver
