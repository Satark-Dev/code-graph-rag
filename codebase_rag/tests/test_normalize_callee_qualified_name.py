from __future__ import annotations

from codebase_rag.parsers.utils import (
    collapse_consecutive_duplicate_qualifier_segments,
    normalize_callee_qualified_name_for_graph,
    normalize_java_trailing_parameter_signature,
    resolve_callee_qualified_name_for_graph_edge,
)
from codebase_rag.types_defs import NodeType


def test_normalize_strips_trailing_signature_list() -> None:
    assert (
        normalize_callee_qualified_name_for_graph(
            "proj.pkg.Clazz.method(String,Object)"
        )
        == "proj.pkg.Clazz.method"
    )


def test_normalize_strips_empty_params() -> None:
    assert normalize_callee_qualified_name_for_graph("ns.Type.foo()") == "ns.Type.foo"


def test_normalize_leaves_ids_without_trailing_signature() -> None:
    assert (
        normalize_callee_qualified_name_for_graph("Function.builtin.Math.round")
        == "Function.builtin.Math.round"
    )


def test_normalize_leaves_unbalanced_or_no_close() -> None:
    assert normalize_callee_qualified_name_for_graph("a.b(") == "a.b("


def test_resolve_edge_prefers_exact_registry_key() -> None:
    registry = {
        "pkg.Clazz.m(String)": NodeType.METHOD,
    }
    assert (
        resolve_callee_qualified_name_for_graph_edge("pkg.Clazz.m(String)", registry)
        == "pkg.Clazz.m(String)"
    )


def test_resolve_edge_falls_back_to_stripped_when_only_base_in_registry() -> None:
    registry = {
        "pkg.Clazz.m": NodeType.METHOD,
    }
    assert (
        resolve_callee_qualified_name_for_graph_edge("pkg.Clazz.m(String)", registry)
        == "pkg.Clazz.m"
    )


def test_collapse_duplicate_consecutive_segments() -> None:
    assert (
        collapse_consecutive_duplicate_qualifier_segments(
            "a.b.UserSessionData.UserSessionData.getValue(String)"
        )
        == "a.b.UserSessionData.getValue(String)"
    )


def test_resolve_edge_uses_collapsed_when_present_in_registry() -> None:
    registry = {
        "x.UserSessionData.getValue(String)": NodeType.METHOD,
    }
    assert (
        resolve_callee_qualified_name_for_graph_edge(
            "x.UserSessionData.UserSessionData.getValue(String)", registry
        )
        == "x.UserSessionData.getValue(String)"
    )


def test_normalize_java_trailing_parameter_signature_simple_types() -> None:
    assert (
        normalize_java_trailing_parameter_signature(
            "pkg.Clazz.getValue(java.lang.String)"
        )
        == "pkg.Clazz.getValue(String)"
    )


def test_resolve_edge_matches_java_lang_vs_simple_name() -> None:
    registry = {
        "pkg.Clazz.getValue(String)": NodeType.METHOD,
    }
    assert (
        resolve_callee_qualified_name_for_graph_edge(
            "pkg.Clazz.getValue(java.lang.String)", registry
        )
        == "pkg.Clazz.getValue(String)"
    )
