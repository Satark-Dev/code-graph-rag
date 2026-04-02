import asyncio
import time
from typing import Any
from uuid import uuid4

from loguru import logger

from ... import constants as cs
from ...config import load_cgrignore_patterns, settings
from ...graph_updater import GraphUpdater
from ...observability.hook import observability_hook
from ...parser_loader import load_parsers
from ...request_context import org_id_context
from ...utils.org_tool_finding_store import (
    get_all_child_findings_for_branch_asset,
    get_branch_name_for_index_asset,
)
from .stage_job_payloads import EvidenceJobPayloadV1
from .producer import kafka_service
from .index_job_payload import IndexJobPayload
from .repo_manager import RepoManager


async def process_index_job_message(
    *,
    payload: IndexJobPayload,
    ingestor: Any,
    repo_manager: RepoManager,
    key: str | None,
) -> bool:
    """
    Returns True if the message offset may be committed (success or poison skip).
    Returns False to leave the offset uncommitted (retry after rebalance/restart).
    """
    invocation = payload.invocation_id
    target_repo_path = None
    ctx_token = org_id_context.set(payload.org_id)
    t0 = time.perf_counter()

    try:
        await observability_hook.before_chat(org_id=payload.org_id, invocation_id=invocation)
        await observability_hook.log_tool_start(tool_name="index", tool_call_id=invocation)

        # Backend sends the branch asset row id; branch = that row's name (type='asset').
        branch_name = get_branch_name_for_index_asset(
            payload.org_tool_findings_id,
            payload.org_id,
        )
        if not branch_name or not str(branch_name).strip():
            logger.error(
                "Index job cannot proceed: missing branch name for asset {} (org_id={}, repo_url={}, invocation_id={})",
                payload.org_tool_findings_id,
                payload.org_id,
                payload.repo_url,
                invocation,
            )
            return True  # Skip poison pill: input is incomplete

        # Ensuring repository exists in the deterministic process folder
        target_repo_path = await repo_manager.ensure_cloned(
            repo_url=payload.repo_url,
            org_id=payload.org_id,
            branch=branch_name,
        )

        try:
            from pathlib import Path

            repo = Path(target_repo_path).resolve()
            cgrignore = load_cgrignore_patterns(repo)
            cli_excludes = frozenset(payload.exclude or [])
            exclude_paths = cli_excludes | cgrignore.exclude or None
            unignore_paths = cgrignore.unignore or None

            if payload.clean:
                logger.info("Cleaning graph database and hash cache before index job...")
                ingestor.clean_database()
                cache_path = repo / cs.HASH_CACHE_FILENAME
                cache_path.unlink(missing_ok=True)

            ingestor.ensure_constraints()
            parsers, queries = load_parsers()

            updater = GraphUpdater(
                ingestor=ingestor,
                repo_path=repo,
                parsers=parsers,
                queries=queries,
                unignore_paths=unignore_paths,
                exclude_paths=exclude_paths,
                project_name=repo.name,
            )

            # Run synchronous GraphUpdater in a thread to avoid blocking the event loop.
            await asyncio.to_thread(updater.run)
            logger.info(
                "Index job completed org_id={} repo_url={} clean={} exclude={} invocation_id={}",
                payload.org_id,
                payload.repo_url,
                payload.clean,
                payload.exclude,
                invocation,
            )
            await observability_hook.log_tool_completed(
                tool_name="index",
                tool_call_id=invocation,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            await observability_hook.log_tool_usage(
                tool_name="index",
                tool_call_id=invocation,
                model_name="non-llm",
                input_tokens=0,
                output_tokens=0,
            )
            await observability_hook.log_message(
                actor="system",
                content=(
                    f"Index completed repo_url={payload.repo_url} "
                    f"clean={payload.clean} exclude={payload.exclude}"
                ),
                tool_call_id=invocation,
            )
            # After successful indexing, enqueue one evidence job per finding for this branch
            # (children of the same branch asset). This ensures evidence/scoring/remediation
            # run separately for each finding.
            try:
                all_finding_ids = get_all_child_findings_for_branch_asset(
                    payload.org_tool_findings_id,
                    payload.org_id,
                )
                
                import os
                if os.environ.get("CGR_KAFKA_SINGLE_FINDING", "0").strip().lower() in ("1", "true", "yes"):
                    all_finding_ids = all_finding_ids[:1]
                    logger.info("CGR_KAFKA_SINGLE_FINDING is enabled; restricting to 1 finding.")

                await kafka_service.start()
                for fid in all_finding_ids:
                    # Create a new invocation id for the scoring pipeline (evidence + scoring + remediation)
                    # so observability can display it as a separate "scoring agent" invocation.
                    scoring_invocation_id = uuid4().hex
                    evidence_payload = EvidenceJobPayloadV1(
                        org_id=payload.org_id,
                        org_tool_findings_ids=[fid],
                        invocation_id=scoring_invocation_id,
                    )
                    await kafka_service.send(
                        settings.KAFKA_EVIDENCE_JOBS_TOPIC,
                        value=evidence_payload.model_dump(mode="json"),
                        key=scoring_invocation_id,
                        raise_on_error=True,
                    )
                logger.info(
                    "Enqueued {} evidence job(s) topic={} org_id={} invocation_id={}",
                    len(all_finding_ids),
                    settings.KAFKA_EVIDENCE_JOBS_TOPIC,
                    payload.org_id,
                    invocation,
                )
            except Exception:
                logger.exception(
                    "Index job {} org_id={}: failed to enqueue evidence job; keeping index result",
                    payload.repo_url,
                    payload.org_id,
                )

            return True
        except Exception:
            logger.exception("Index job failed for {} (invocation={})", payload.repo_url, invocation)
            await observability_hook.log_tool_failed(
                tool_name="index",
                tool_call_id=invocation,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            return False

    except Exception as e:
        logger.error("Failed to prepare repository for indexing: {}", e)
        await observability_hook.log_tool_failed(
            tool_name="index",
            tool_call_id=invocation,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
        return True  # Skip poison pill
    finally:
        org_id_context.reset(ctx_token)

