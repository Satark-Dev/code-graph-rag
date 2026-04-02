from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from tree_sitter import Node

from ... import constants as cs
from ... import logs as lg
from ...types_defs import (
    FunctionRegistryTrieProtocol,
    LanguageQueries,
    SimpleNameLookup,
)
from ..import_processor import ImportProcessor
from .ast_analyzer import PythonAstAnalyzerMixin
from .expression_analyzer import PythonExpressionAnalyzerMixin
from .variable_analyzer import PythonVariableAnalyzerMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..factory import ASTCacheProtocol
    from ..js_ts import JsTypeInferenceEngine


class PythonTypeInferenceEngine(
    PythonExpressionAnalyzerMixin,
    PythonAstAnalyzerMixin,
    PythonVariableAnalyzerMixin,
):
    __slots__ = (
        "import_processor",
        "function_registry",
        "repo_path",
        "project_name",
        "ast_cache",
        "queries",
        "module_qn_to_file_path",
        "class_inheritance",
        "simple_name_lookup",
        "_js_type_inference_getter",
        "_method_return_type_cache",
        "_class_property_cache",
        "_type_inference_in_progress",
    )

    def __init__(
        self,
        import_processor: ImportProcessor,
        function_registry: FunctionRegistryTrieProtocol,
        repo_path: Path,
        project_name: str,
        ast_cache: ASTCacheProtocol,
        queries: dict[cs.SupportedLanguage, LanguageQueries],
        module_qn_to_file_path: dict[str, Path],
        class_inheritance: dict[str, list[str]],
        simple_name_lookup: SimpleNameLookup,
        js_type_inference_getter: Callable[[], JsTypeInferenceEngine],
    ):
        self.import_processor = import_processor
        self.function_registry = function_registry
        self.repo_path = repo_path
        self.project_name = project_name
        self.ast_cache = ast_cache
        self.queries = queries
        self.module_qn_to_file_path = module_qn_to_file_path
        self.class_inheritance = class_inheritance
        self.simple_name_lookup = simple_name_lookup
        self._js_type_inference_getter = js_type_inference_getter

        self._method_return_type_cache: dict[str, str | None] = {}
        self._class_property_cache: dict[str, dict[str, str]] = {}
        self._type_inference_in_progress: set[str] = set()

    def build_local_variable_type_map(
        self, caller_node: Node, module_qn: str, class_context: str | None = None
    ) -> dict[str, str]:
        local_var_types: dict[str, str] = {}

        if class_context:
            local_var_types.update(self._get_class_property_map(class_context, module_qn))

        try:
            self._infer_parameter_types(caller_node, local_var_types, module_qn)
            # (H) Single-pass traversal avoids O(5*N) multiple traversals for type inference.
            self._traverse_single_pass(caller_node, local_var_types, module_qn)

        except Exception as e:
            logger.debug(lg.PY_BUILD_VAR_MAP_FAILED, error=e)

        return local_var_types

    def _get_class_property_map(self, class_qn: str, module_qn: str) -> dict[str, str]:
        if class_qn in self._class_property_cache:
            return self._class_property_cache[class_qn]

        if class_qn in self._type_inference_in_progress:
            return {}

        self._type_inference_in_progress.add(class_qn)
        try:
            class_node = self._find_class_node(class_qn)
            if not class_node:
                return {}

            property_map: dict[str, str] = {}
            # 1. Fetch class-level declarations (e.g. db: Database)
            self._analyze_class_level_declarations(class_node, property_map, module_qn)
            # 2. Fetch all self.attr assignments within all methods of this class
            self._analyze_self_assignments(class_node, property_map, module_qn)

            self._class_property_cache[class_qn] = property_map
            return property_map
        finally:
            self._type_inference_in_progress.remove(class_qn)
