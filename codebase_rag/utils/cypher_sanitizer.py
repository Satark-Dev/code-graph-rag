import re

_CYPHER_LINE_COMMENT = re.compile(r"//[^\n]*")
_CYPHER_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CYPHER_LABEL_UNION = re.compile(
    r":\s*\(\s*\w+\s*\|\s*\w+.*?\)|:\s*\w+\s*\|\s*\w+",
    re.IGNORECASE,
)
_CYPHER_REL_TYPE_UNION = re.compile(
    r"\[\s*[^]]*:\s*[A-Za-z_]\w*(?:\s*\|\s*[A-Za-z_]\w*)+[^]]*\]",
    re.IGNORECASE,
)
_CYPHER_EXISTS_PROP = re.compile(
    r"EXISTS\s*\(\s*([A-Za-z_][\w]*\.[A-Za-z_][\w]*)\s*\)", re.IGNORECASE
)
_OPTIONAL_MATCH = re.compile(r"^\s*OPTIONAL\s+MATCH\b", re.IGNORECASE | re.MULTILINE)
_MATCH_LINE = re.compile(r"^\s*MATCH\b", re.IGNORECASE)
_BARE_NODE_MATCH_LINE = re.compile(
    r"^\s*MATCH\s*\(\s*[A-Za-z_]\w*\s*\)\s*$", re.IGNORECASE
)
_BARE_NODE_MATCH_ANYWHERE = re.compile(
    r"^\s*MATCH\s*\(\s*[A-Za-z_]\w*\s*\)\s*$", re.IGNORECASE | re.MULTILINE
)
_RETURN_DISTINCT = re.compile(r"\bRETURN\s+DISTINCT\b", re.IGNORECASE)
_RETURN = re.compile(r"\bRETURN\b", re.IGNORECASE)
_TAUTOLOGY_STARTS_WITH = re.compile(
    r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s+STARTS\s+WITH\s+\1\.\2\b",
    re.IGNORECASE,
)
_TAUTOLOGY_ENDS_WITH = re.compile(
    r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s+ENDS\s+WITH\s+\1\.\2\b",
    re.IGNORECASE,
)
_TAUTOLOGY_EQUALS = re.compile(
    r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=\s*\1\.\2\b",
    re.IGNORECASE,
)


class CypherSanitizer:
    """Utility class to sanitize and validate Cypher queries for Memgraph."""

    @staticmethod
    def first_statement(query: str) -> str:
        parts = [part.strip() for part in query.split(";") if part.strip()]
        if not parts:
            return query.strip()
        return parts[0] + ";"

    @staticmethod
    def strip_comments(query: str) -> str:
        no_block = _CYPHER_BLOCK_COMMENT.sub("", query)
        return _CYPHER_LINE_COMMENT.sub("", no_block)

    @staticmethod
    def contains_label_union(query: str) -> bool:
        return _CYPHER_LABEL_UNION.search(query) is not None

    @staticmethod
    def contains_relationship_type_union(query: str) -> bool:
        return _CYPHER_REL_TYPE_UNION.search(query) is not None

    @staticmethod
    def contains_union_syntax(query: str) -> bool:
        """
        Memgraph doesn't accept label unions (e.g. :A|B) or relationship-type unions (e.g. [:A|B]).
        """
        return CypherSanitizer.contains_label_union(
            query
        ) or CypherSanitizer.contains_relationship_type_union(query)

    @staticmethod
    def contains_property_exists(query: str) -> bool:
        return _CYPHER_EXISTS_PROP.search(query) is not None

    @staticmethod
    def replace_property_exists(query: str) -> str:
        return _CYPHER_EXISTS_PROP.sub(lambda m: f"{m.group(1)} IS NOT NULL", query)

    @staticmethod
    def contains_match_after_optional(query: str) -> bool:
        """
        Memgraph rejects plain MATCH clauses after an OPTIONAL MATCH.
        """
        lines = query.splitlines()
        seen_optional = False
        for line in lines:
            if _OPTIONAL_MATCH.match(line):
                seen_optional = True
                continue
            if seen_optional and _MATCH_LINE.match(line) and not _OPTIONAL_MATCH.match(line):
                return True
        return False

    @staticmethod
    def rewrite_match_after_optional(query: str) -> str:
        """
        Rewrite plain MATCH clauses occurring after the first OPTIONAL MATCH to OPTIONAL MATCH.
        This preserves semantics reasonably well for the "LLM-generated exploratory" queries
        and avoids Memgraph parse errors.
        """
        lines = query.splitlines()
        out: list[str] = []
        seen_optional = False
        for line in lines:
            if _OPTIONAL_MATCH.match(line):
                seen_optional = True
                out.append(line)
                continue
            if seen_optional and _MATCH_LINE.match(line) and not _OPTIONAL_MATCH.match(line):
                # Preserve original indentation.
                leading = line[: len(line) - len(line.lstrip())]
                rest = line.lstrip()[len("MATCH") :]
                out.append(f"{leading}OPTIONAL MATCH{rest}")
                continue
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def contains_bare_node_match(query: str) -> bool:
        """
        Detects standalone 'MATCH (x)' clauses which often create cartesian products
        in LLM-generated Cypher (especially after WITH/OPTIONAL MATCH blocks).
        """
        return _BARE_NODE_MATCH_ANYWHERE.search(query) is not None

    @staticmethod
    def ensure_return_distinct(query: str) -> str:
        """
        Ensure the first RETURN clause is RETURN DISTINCT to reduce duplicates.
        """
        if _RETURN_DISTINCT.search(query):
            return query
        return _RETURN.sub("RETURN DISTINCT", query, count=1)

    @staticmethod
    def rewrite_tautologies(query: str) -> str:
        """
        Replace obviously-true predicates like `f.path STARTS WITH f.path` with `TRUE`.
        This doesn't change semantics but reduces noise and can help downstream heuristics.
        """
        query = _TAUTOLOGY_STARTS_WITH.sub("TRUE", query)
        query = _TAUTOLOGY_ENDS_WITH.sub("TRUE", query)
        query = _TAUTOLOGY_EQUALS.sub("TRUE", query)
        return query

    @staticmethod
    def append_memgraph_constraints(nl_query: str) -> str:
        return (
            f"{nl_query}\n\n"
            "IMPORTANT: Output valid Memgraph Cypher. "
            "Do NOT use line or block comments (// or /* */). "
            "Do NOT use label unions like :A|B or :(A|B). "
            "Do NOT use EXISTS(property). Use `property IS NOT NULL` instead. "
            "Do NOT put plain MATCH after OPTIONAL MATCH; use OPTIONAL MATCH consistently. "
            "Avoid standalone MATCH (n). Do NOT create cartesian products; only match patterns "
            "connected to previously bound variables. Prefer RETURN DISTINCT. "
            "Avoid tautologies like `x.prop STARTS WITH x.prop`. "
            "Return a SINGLE Cypher statement only (no multiple MATCH/RETURN blocks). "
            "If you need an OR across labels, use WHERE (n:A OR n:B). "
            "Use only MATCH/WHERE/RETURN/LIMIT and keep syntax simple."
        )

    @staticmethod
    def append_memgraph_strict_constraints(nl_query: str) -> str:
        return (
            f"{CypherSanitizer.append_memgraph_constraints(nl_query)}\n\n"
            "STRICT MODE: Do NOT use the `|` character anywhere in the Cypher output "
            "(no label unions and no relationship type unions)."
        )
