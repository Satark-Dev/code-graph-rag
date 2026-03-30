from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from loguru import logger
from rich.prompt import Prompt
from rich.table import Table

from . import constants as cs
from . import logs as ls
from .config import load_cgrignore_patterns
from .context import app_context
from .services.graph_service import MemgraphIngestor
from .types_defs import GraphData
from .ui import style


def _write_graph_json(ingestor: MemgraphIngestor, output_path: Path) -> GraphData:
    graph_data: GraphData = ingestor.export_graph_to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding=cs.ENCODING_UTF8) as f:
        json.dump(graph_data, f, indent=cs.JSON_INDENT, ensure_ascii=False)

    return graph_data


def export_graph_to_file(ingestor: MemgraphIngestor, output: str) -> bool:
    output_path = Path(output)

    try:
        graph_data = _write_graph_json(ingestor, output_path)
        metadata = graph_data[cs.KEY_METADATA]
        app_context.console.print(
            cs.UI_GRAPH_EXPORT_SUCCESS.format(path=output_path.absolute())
        )
        app_context.console.print(
            cs.UI_GRAPH_EXPORT_STATS.format(
                nodes=metadata[cs.KEY_TOTAL_NODES],
                relationships=metadata[cs.KEY_TOTAL_RELATIONSHIPS],
            )
        )
        return True

    except Exception as e:  # noqa: BLE001
        app_context.console.print(cs.UI_ERR_EXPORT_FAILED.format(error=e))
        logger.exception(ls.EXPORT_ERROR.format(error=e))
        return False


def detect_excludable_directories(repo_path: Path) -> set[str]:
    detected: set[str] = set()
    queue: deque[tuple[Path, int]] = deque([(repo_path, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth > cs.INTERACTIVE_BFS_MAX_DEPTH:
            continue
        try:
            entries = list(current.iterdir())
        except PermissionError:
            continue
        for path in entries:
            if not path.is_dir():
                continue
            if path.name in cs.IGNORE_PATTERNS:
                detected.add(path.relative_to(repo_path).as_posix())
            else:
                queue.append((path, depth + 1))
    return detected


def _get_grouping_key(path: str) -> str:
    parts = Path(path).parts
    if not parts:
        return cs.INTERACTIVE_DEFAULT_GROUP
    for part in parts:
        if part in cs.IGNORE_PATTERNS:
            return part
    return parts[0]


def _group_paths_by_pattern(paths: set[str]) -> dict[str, list[str]]:
    from collections import defaultdict

    groups: defaultdict[str, list[str]] = defaultdict(list)
    for path in paths:
        key = _get_grouping_key(path)
        groups[key].append(path)
    for group_paths in groups.values():
        group_paths.sort()
    return groups


def _format_nested_count(count: int) -> str:
    template = (
        cs.INTERACTIVE_NESTED_SINGULAR if count == 1 else cs.INTERACTIVE_NESTED_PLURAL
    )
    return template.format(count=count)


def _display_grouped_table(groups: dict[str, list[str]]) -> list[str]:
    sorted_roots = sorted(groups.keys())
    table = Table(title=style(cs.INTERACTIVE_TITLE_GROUPED, cs.Color.CYAN))
    table.add_column(cs.INTERACTIVE_COL_NUM, style=cs.Color.YELLOW, width=4)
    table.add_column(cs.INTERACTIVE_COL_PATTERN)
    table.add_column(cs.INTERACTIVE_COL_NESTED, style=cs.INTERACTIVE_STYLE_DIM)

    for i, root in enumerate(sorted_roots, 1):
        nested_count = len(groups[root])
        table.add_row(str(i), root, _format_nested_count(nested_count))

    app_context.console.print(table)
    app_context.console.print(
        style(
            cs.INTERACTIVE_INSTRUCTIONS_GROUPED, cs.Color.YELLOW, cs.StyleModifier.NONE
        )
    )
    return sorted_roots


def _display_nested_table(pattern: str, paths: list[str]) -> None:
    title = cs.INTERACTIVE_TITLE_NESTED.format(pattern=pattern)
    table = Table(title=style(title, cs.Color.CYAN))
    table.add_column(cs.INTERACTIVE_COL_NUM, style=cs.Color.YELLOW, width=4)
    table.add_column(cs.INTERACTIVE_COL_PATH)

    for i, path in enumerate(paths, 1):
        table.add_row(str(i), path)

    app_context.console.print(table)
    app_context.console.print(
        style(
            cs.INTERACTIVE_INSTRUCTIONS_NESTED.format(pattern=pattern),
            cs.Color.YELLOW,
            cs.StyleModifier.NONE,
        )
    )


def _prompt_nested_selection(pattern: str, paths: list[str]) -> set[str]:
    _display_nested_table(pattern, paths)

    response = Prompt.ask(
        style(cs.INTERACTIVE_PROMPT_KEEP, cs.Color.CYAN),
        default=cs.INTERACTIVE_KEEP_NONE,
    )

    if response.lower() == cs.INTERACTIVE_KEEP_ALL:
        return set(paths)
    if response.lower() == cs.INTERACTIVE_KEEP_NONE:
        return set()

    selected: set[str] = set()
    for part in response.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(paths):
                selected.add(paths[idx])
            else:
                logger.warning(ls.EXCLUDE_INVALID_INDEX.format(index=part))
        else:
            logger.warning(ls.EXCLUDE_INVALID_INPUT.format(input=part))

    return selected


def prompt_for_unignored_directories(
    repo_path: Path,
    cli_excludes: list[str] | None = None,
) -> frozenset[str]:
    detected = detect_excludable_directories(repo_path)
    cgrignore = load_cgrignore_patterns(repo_path)
    cli_patterns = frozenset(cli_excludes) if cli_excludes else frozenset()
    pre_excluded = cli_patterns | cgrignore.exclude

    if not detected and not pre_excluded:
        return cgrignore.unignore

    all_candidates = detected | pre_excluded
    groups = _group_paths_by_pattern(all_candidates)
    sorted_roots = _display_grouped_table(groups)

    response = Prompt.ask(
        style(cs.INTERACTIVE_PROMPT_KEEP, cs.Color.CYAN),
        default=cs.INTERACTIVE_KEEP_NONE,
    )

    if response.lower() == cs.INTERACTIVE_KEEP_ALL:
        return frozenset(all_candidates) | cgrignore.unignore

    if response.lower() == cs.INTERACTIVE_KEEP_NONE:
        return cgrignore.unignore

    selected: set[str] = set()
    expand_requests: list[int] = []
    regular_selections: list[int] = []

    for part in response.split(","):
        part = part.strip().lower()
        if not part:
            continue

        if part.endswith(cs.INTERACTIVE_EXPAND_SUFFIX) and part[:-1].isdigit():
            expand_requests.append(int(part[:-1]) - 1)
        elif part.isdigit():
            regular_selections.append(int(part) - 1)
        else:
            logger.warning(ls.EXCLUDE_INVALID_INPUT.format(input=part))

    for idx in expand_requests:
        if 0 <= idx < len(sorted_roots):
            root = sorted_roots[idx]
            nested_selected = _prompt_nested_selection(root, groups[root])
            selected.update(nested_selected)
        else:
            logger.warning(ls.EXCLUDE_INVALID_INDEX.format(index=idx + 1))

    for idx in regular_selections:
        if 0 <= idx < len(sorted_roots):
            root = sorted_roots[idx]
            selected.update(groups[root])
        else:
            logger.warning(ls.EXCLUDE_INVALID_INDEX.format(index=idx + 1))

    return frozenset(selected) | cgrignore.unignore

