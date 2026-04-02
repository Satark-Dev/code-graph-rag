from __future__ import annotations

from codebase_rag.utils.cypher_sanitizer import CypherSanitizer


def test_rewrite_match_after_optional_converts_late_match() -> None:
    q = "\n".join(
        [
            "MATCH (f:File)",
            "OPTIONAL MATCH (f)-[:CONTAINS]->(m:Module)",
            "OPTIONAL MATCH (m)-[:DEFINES]->(fn:Function)",
            "MATCH (n)",
            "WHERE n.name IS NOT NULL",
            "RETURN n.name",
            "LIMIT 1;",
        ]
    )
    assert CypherSanitizer.contains_match_after_optional(q) is True
    rewritten = CypherSanitizer.rewrite_match_after_optional(q)
    assert "OPTIONAL MATCH (n)" in rewritten
    assert CypherSanitizer.contains_match_after_optional(rewritten) is False


def test_contains_bare_node_match_detects_match_n() -> None:
    q = "MATCH (n)\nRETURN n;"
    assert CypherSanitizer.contains_bare_node_match(q) is True


def test_contains_bare_node_match_detects_any_var() -> None:
    q = "MATCH (x)\nWHERE x.name IS NOT NULL\nMATCH (x2)\nRETURN x2;"
    assert CypherSanitizer.contains_bare_node_match(q) is True


def test_ensure_return_distinct_adds_distinct() -> None:
    q = "MATCH (n:File)\nRETURN n.path AS path\nLIMIT 5;"
    rewritten = CypherSanitizer.ensure_return_distinct(q)
    assert "RETURN DISTINCT" in rewritten


def test_rewrite_tautologies_replaces_startswith_self() -> None:
    q = "MATCH (f:File)\nWHERE f.path STARTS WITH f.path\nRETURN f.path;"
    rewritten = CypherSanitizer.rewrite_tautologies(q)
    assert "WHERE TRUE" in rewritten


def test_contains_relationship_type_union_detects_union() -> None:
    q = "MATCH (a)-[:CONTAINS_FILE|CONTAINS_MODULE*]->(b) RETURN a;"
    assert CypherSanitizer.contains_relationship_type_union(q) is True
    assert CypherSanitizer.contains_union_syntax(q) is True


def test_contains_relationship_type_union_ignores_single_type() -> None:
    q = "MATCH (a)-[:CONTAINS_FILE*]->(b) RETURN a;"
    assert CypherSanitizer.contains_relationship_type_union(q) is False
    assert CypherSanitizer.contains_union_syntax(q) is False

