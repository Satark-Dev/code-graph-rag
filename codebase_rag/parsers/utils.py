from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger
from tree_sitter import Node, Query, QueryCursor

from .. import constants as cs
from .. import logs
from ..types_defs import (
    ASTNode,
    FunctionRegistryTrieProtocol,
    LanguageQueries,
    NodeType,
    PropertyDict,
    TreeSitterNodeProtocol,
)

if TYPE_CHECKING:
    from ..language_spec import LanguageSpec
    from ..services import IngestorProtocol
    from ..types_defs import FunctionRegistryTrieProtocol

from ..types_defs import SimpleNameLookup


class FunctionCapturesResult(NamedTuple):
    lang_config: LanguageSpec
    captures: dict[str, list[ASTNode]]


def sorted_captures(cursor: QueryCursor, node: ASTNode) -> dict[str, list[ASTNode]]:
    # (H) tree-sitter v0.25 captures() returns nodes in non-deterministic order
    # across process invocations; sort by start_byte for reproducibility
    raw = cursor.captures(node)
    return {
        name: sorted(nodes, key=lambda n: n.start_byte) for name, nodes in raw.items()
    }


def get_function_captures(
    root_node: ASTNode,
    language: cs.SupportedLanguage,
    queries: dict[cs.SupportedLanguage, LanguageQueries],
) -> FunctionCapturesResult | None:
    lang_queries = queries[language]
    lang_config = lang_queries[cs.QUERY_CONFIG]

    if not (query := lang_queries[cs.QUERY_FUNCTIONS]):
        return None

    cursor = QueryCursor(query)
    captures = sorted_captures(cursor, root_node)
    return FunctionCapturesResult(lang_config, captures)


@lru_cache(maxsize=50000)
def _cached_decode_bytes(text_bytes: bytes) -> str:
    return text_bytes.decode(cs.ENCODING_UTF8)


def safe_decode_text(node: ASTNode | TreeSitterNodeProtocol | None) -> str | None:
    if node is None or (text_bytes := node.text) is None:
        return None
    if isinstance(text_bytes, bytes):
        return _cached_decode_bytes(text_bytes)
    return str(text_bytes)


def get_query_cursor(query: Query) -> QueryCursor:
    return QueryCursor(query)


def safe_decode_with_fallback(node: ASTNode | None, fallback: str = "") -> str:
    return result if (result := safe_decode_text(node)) is not None else fallback


def normalize_callee_qualified_name_for_graph(qualified_name: str) -> str:
    """
    Strip a trailing balanced ``(...)`` overload / parameter suffix from a symbol path.

    Some resolvers emit overload-qualified names while graph nodes use the base path;
    others (e.g. Java method ingestion) store the parameter list in ``qualified_name``.
    For ``CALLS`` edges, use :func:`resolve_callee_qualified_name_for_graph_edge` so
    the chosen id matches whatever is actually present in the registry.
    """
    if not qualified_name or "(" not in qualified_name:
        return qualified_name
    s = qualified_name.rstrip()
    if not s.endswith(")"):
        return qualified_name
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ")":
            depth += 1
        elif s[i] == "(":
            depth -= 1
            if depth == 0:
                return s[:i]
    return qualified_name


def normalize_java_trailing_parameter_signature(qualified_name: str) -> str:
    """
    Normalize a Java method's trailing ``(...)`` so registry keys match across
    ``String`` vs ``java.lang.String`` style AST text.
    """
    if "(" not in qualified_name or not qualified_name.endswith(")"):
        return qualified_name
    open_idx = qualified_name.rfind("(")
    if open_idx == -1:
        return qualified_name
    prefix = qualified_name[:open_idx]
    inner = qualified_name[open_idx + 1 : -1].strip()
    if not inner:
        return f"{prefix}{cs.EMPTY_PARENS}"
    parts: list[str] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        parts.append(part.rsplit(cs.SEPARATOR_DOT, 1)[-1])
    return f"{prefix}({','.join(parts)})"


def _java_method_segment_suffix(qualified_name: str) -> str | None:
    """Last path segment including ``(`` and parameters, e.g. ``getValue(String)``."""
    if "(" not in qualified_name:
        return None
    idx = qualified_name.rfind(cs.SEPARATOR_DOT)
    if idx == -1:
        return qualified_name
    return qualified_name[idx + 1 :]


def _pick_best_java_method_match(reference: str, matches: list[str]) -> str | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    ref_variants = _unique_ordered_java_qn_variants(reference)
    for pref in ref_variants:
        for m in matches:
            if m == pref:
                return m
    best = matches[0]
    best_len = 0
    ref_parts = reference.split(cs.SEPARATOR_DOT)
    for m in matches:
        mp = m.split(cs.SEPARATOR_DOT)
        n = 0
        for i in range(min(len(ref_parts), len(mp))):
            if ref_parts[i] == mp[i]:
                n += 1
            else:
                break
        if n > best_len:
            best_len = n
            best = m
    return best


def _unique_ordered_java_qn_variants(qn: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in (
        qn,
        normalize_java_trailing_parameter_signature(qn),
        collapse_consecutive_duplicate_qualifier_segments(qn),
        collapse_consecutive_duplicate_qualifier_segments(
            normalize_java_trailing_parameter_signature(qn)
        ),
        normalize_callee_qualified_name_for_graph(qn),
        normalize_callee_qualified_name_for_graph(
            normalize_java_trailing_parameter_signature(qn)
        ),
    ):
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _registry_keys_ending_with(
    registry: FunctionRegistryTrieProtocol, suffix: str
) -> list[str]:
    if hasattr(registry, "find_ending_with"):
        return list(registry.find_ending_with(suffix))
    keys = registry.keys() if hasattr(registry, "keys") else registry  # type: ignore[arg-type]
    return sorted(k for k in keys if str(k).endswith(suffix))


def _resolve_java_like_qn_to_registry(
    qn: str,
    registry: FunctionRegistryTrieProtocol,
) -> str | None:
    for candidate in _unique_ordered_java_qn_variants(qn):
        if candidate in registry:
            return candidate
    collapsed_norm = normalize_callee_qualified_name_for_graph(
        collapse_consecutive_duplicate_qualifier_segments(qn)
    )
    if collapsed_norm in registry:
        return collapsed_norm
    seg = _java_method_segment_suffix(qn)
    if not seg or "(" not in seg:
        return None
    matches = _registry_keys_ending_with(registry, seg)
    if not matches:
        nj = normalize_java_trailing_parameter_signature(qn)
        if nj != qn:
            seg2 = _java_method_segment_suffix(nj)
            if seg2 and seg2 != seg:
                matches = _registry_keys_ending_with(registry, seg2)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    picked = _pick_best_java_method_match(qn, matches)
    return picked


def collapse_consecutive_duplicate_qualifier_segments(qualified_name: str) -> str:
    """
    Drop immediately repeated path segments (e.g. ``Foo.Foo.bar`` -> ``Foo.bar``).

    Resolvers sometimes double the same class/simple name when nesting is inferred
    incorrectly; ingested nodes typically use a single segment.
    """
    parts = qualified_name.split(cs.SEPARATOR_DOT)
    if len(parts) < 2:
        return qualified_name
    out: list[str] = [parts[0]]
    for p in parts[1:]:
        if out and out[-1] == p:
            continue
        out.append(p)
    return cs.SEPARATOR_DOT.join(out)


def resolve_callee_qualified_name_for_graph_edge(
    callee_qn: str,
    registry: FunctionRegistryTrieProtocol,
) -> str:
    """
    Pick a callee ``qualified_name`` that matches an ingested registry key when possible.

    Handles Java-style overload keys, ``java.lang.String`` vs ``String``, duplicate
    path segments, and trie ``find_ending_with`` disambiguation when needed.
    """
    if resolved := _resolve_java_like_qn_to_registry(callee_qn, registry):
        return resolved
    return callee_qn


def resolve_caller_qualified_name_for_graph_edge(
    caller_qn: str,
    registry: FunctionRegistryTrieProtocol,
) -> str:
    """Align caller id with registry (Java callers include ``(Type...)`` like ingested nodes)."""
    if resolved := _resolve_java_like_qn_to_registry(caller_qn, registry):
        return resolved
    return caller_qn


def contains_node(parent: ASTNode, target: ASTNode) -> bool:
    return parent == target or any(
        contains_node(child, target) for child in parent.children
    )


def ingest_method(
    method_node: ASTNode,
    container_qn: str,
    container_type: cs.NodeLabel,
    ingestor: IngestorProtocol,
    function_registry: FunctionRegistryTrieProtocol,
    simple_name_lookup: SimpleNameLookup,
    get_docstring_func: Callable[[ASTNode], str | None],
    language: cs.SupportedLanguage | None = None,
    extract_decorators_func: Callable[[ASTNode], list[str]] | None = None,
    method_qualified_name: str | None = None,
    file_path: Path | None = None,
    repo_path: Path | None = None,
) -> None:
    if language == cs.SupportedLanguage.CPP:
        from .cpp import utils as cpp_utils

        method_name = cpp_utils.extract_function_name(method_node)
        if not method_name:
            return
    elif not (method_name_node := method_node.child_by_field_name(cs.FIELD_NAME)):
        return
    elif (text := method_name_node.text) is None:
        return
    else:
        method_name = text.decode(cs.ENCODING_UTF8)

    method_qn = method_qualified_name or f"{container_qn}.{method_name}"

    decorators = extract_decorators_func(method_node) if extract_decorators_func else []

    method_props: PropertyDict = {
        cs.KEY_QUALIFIED_NAME: method_qn,
        cs.KEY_NAME: method_name,
        cs.KEY_DECORATORS: decorators,
        cs.KEY_START_LINE: method_node.start_point[0] + 1,
        cs.KEY_END_LINE: method_node.end_point[0] + 1,
        cs.KEY_DOCSTRING: get_docstring_func(method_node),
    }
    if file_path is not None and repo_path is not None:
        method_props[cs.KEY_PATH] = file_path.relative_to(repo_path).as_posix()
        method_props[cs.KEY_ABSOLUTE_PATH] = file_path.resolve().as_posix()

    logger.info(logs.METHOD_FOUND.format(name=method_name, qn=method_qn))
    ingestor.ensure_node_batch(cs.NodeLabel.METHOD, method_props)
    function_registry[method_qn] = NodeType.METHOD
    simple_name_lookup[method_name].add(method_qn)

    ingestor.ensure_relationship_batch(
        (container_type, cs.KEY_QUALIFIED_NAME, container_qn),
        cs.RelationshipType.DEFINES_METHOD,
        (cs.NodeLabel.METHOD, cs.KEY_QUALIFIED_NAME, method_qn),
    )


def ingest_exported_function(
    function_node: ASTNode,
    function_name: str,
    module_qn: str,
    export_type: str,
    ingestor: IngestorProtocol,
    function_registry: FunctionRegistryTrieProtocol,
    simple_name_lookup: SimpleNameLookup,
    get_docstring_func: Callable[[ASTNode], str | None],
    is_export_inside_function_func: Callable[[ASTNode], bool],
) -> None:
    if is_export_inside_function_func(function_node):
        return

    function_qn = f"{module_qn}.{function_name}"

    function_props = {
        cs.KEY_QUALIFIED_NAME: function_qn,
        cs.KEY_NAME: function_name,
        cs.KEY_START_LINE: function_node.start_point[0] + 1,
        cs.KEY_END_LINE: function_node.end_point[0] + 1,
        cs.KEY_DOCSTRING: get_docstring_func(function_node),
    }

    logger.info(
        logs.EXPORT_FOUND.format(
            export_type=export_type, name=function_name, qn=function_qn
        )
    )
    ingestor.ensure_node_batch(cs.NodeLabel.FUNCTION, function_props)
    function_registry[function_qn] = NodeType.FUNCTION
    simple_name_lookup[function_name].add(function_qn)


def is_method_node(func_node: ASTNode, lang_config: LanguageSpec) -> bool:
    current = func_node.parent
    if not isinstance(current, Node):
        return False

    while current and current.type not in lang_config.module_node_types:
        if (
            current.type in lang_config.function_node_types
            and current.child_by_field_name(cs.FIELD_BODY) is not None
        ):
            return False
        if current.type in lang_config.class_node_types:
            return True
        current = current.parent
    return False
