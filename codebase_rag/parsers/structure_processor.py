from pathlib import Path

from loguru import logger

from .. import constants as cs
from .. import logs
from ..services import IngestorProtocol
from ..types_defs import LanguageQueries, NodeIdentifier
from ..utils.path_utils import should_skip_path


class StructureProcessor:
    __slots__ = (
        "ingestor",
        "repo_path",
        "project_name",
        "queries",
        "structural_elements",
        "unignore_paths",
        "exclude_paths",
    )

    def __init__(
        self,
        ingestor: IngestorProtocol,
        repo_path: Path,
        project_name: str,
        queries: dict[cs.SupportedLanguage, LanguageQueries],
        unignore_paths: frozenset[str] | None = None,
        exclude_paths: frozenset[str] | None = None,
    ):
        self.ingestor = ingestor
        self.repo_path = repo_path
        self.project_name = project_name
        self.queries = queries
        self.structural_elements: dict[Path, str | None] = {}
        self.unignore_paths = unignore_paths
        self.exclude_paths = exclude_paths

    def _get_parent_identifier(
        self, parent_rel_path: Path, parent_container_qn: str | None
    ) -> NodeIdentifier:
        if parent_rel_path == Path(cs.PATH_CURRENT_DIR):
            return (cs.NodeLabel.PROJECT, cs.KEY_NAME, self.project_name)
        if parent_container_qn:
            return (cs.NodeLabel.PACKAGE, cs.KEY_QUALIFIED_NAME, parent_container_qn)
        return (cs.NodeLabel.FOLDER, cs.KEY_PATH, parent_rel_path.as_posix())

    def _should_keep_dir(self, dirname: str, dir_prefix: str) -> bool:
        if dirname not in cs.IGNORE_PATTERNS and (
            not self.exclude_paths or dirname not in self.exclude_paths
        ):
            return True
        return bool(
            self.unignore_paths
            and any(
                u.startswith(f"{dir_prefix}{dirname}/") or u == f"{dir_prefix}{dirname}"
                for u in self.unignore_paths
            )
        )

    def identify_structure(self, collect_files: bool = False) -> list[Path]:
        eligible_files: list[Path] = []
        # We start with the repo root. Note that os.walk will yield it first.
        directories = {self.repo_path}
        package_indicators: set[str] = set()

        for lang_queries in self.queries.values():
            lang_config = lang_queries[cs.QUERY_CONFIG]
            package_indicators.update(lang_config.package_indicators)

        # Map to store if a directory is a package, populated during traversal
        is_package_dir: dict[Path, bool] = {}

        import os

        for dirpath, dirnames, filenames in os.walk(str(self.repo_path)):
            root = Path(dirpath)
            relative_root = root.relative_to(self.repo_path)
            rel_dir_str = relative_root.as_posix()
            dir_prefix = "" if rel_dir_str == "." else f"{rel_dir_str}/"

            # 1. Identify if this root is a package
            is_package_dir[root] = any(f in package_indicators for f in filenames)

            # 2. Prune directories for the walk
            dirnames[:] = sorted(
                [d for d in dirnames if self._should_keep_dir(d, dir_prefix)]
            )
            for d in dirnames:
                directories.add(root / d)

            # 3. Optionally collect files (Pass 2 consolidation)
            if collect_files:
                for fname in sorted(filenames):
                    if fname == cs.HASH_CACHE_FILENAME:
                        continue
                    filepath = root / fname
                    if not should_skip_path(
                        filepath,
                        self.repo_path,
                        exclude_paths=self.exclude_paths,
                        unignore_paths=self.unignore_paths,
                    ):
                        eligible_files.append(filepath)

        # Process the collected directories in order
        for root in sorted(directories):
            relative_root = root.relative_to(self.repo_path)
            parent_rel_path = relative_root.parent
            parent_container_qn = self.structural_elements.get(parent_rel_path)

            if is_package_dir.get(root, False):
                package_qn = cs.SEPARATOR_DOT.join(
                    [self.project_name] + list(relative_root.parts)
                )
                self.structural_elements[relative_root] = package_qn
                logger.info(logs.STRUCT_IDENTIFIED_PACKAGE.format(package_qn=package_qn))
                self.ingestor.ensure_node_batch(
                    cs.NodeLabel.PACKAGE,
                    {
                        cs.KEY_QUALIFIED_NAME: package_qn,
                        cs.KEY_NAME: root.name,
                        cs.KEY_PATH: relative_root.as_posix(),
                        cs.KEY_ABSOLUTE_PATH: root.resolve().as_posix(),
                    },
                )
                parent_identifier = self._get_parent_identifier(
                    parent_rel_path, parent_container_qn
                )
                self.ingestor.ensure_relationship_batch(
                    parent_identifier,
                    cs.RelationshipType.CONTAINS_PACKAGE,
                    (cs.NodeLabel.PACKAGE, cs.KEY_QUALIFIED_NAME, package_qn),
                )
            elif root != self.repo_path:
                self.structural_elements[relative_root] = None
                logger.info(
                    logs.STRUCT_IDENTIFIED_FOLDER.format(relative_root=relative_root)
                )
                self.ingestor.ensure_node_batch(
                    cs.NodeLabel.FOLDER,
                    {
                        cs.KEY_PATH: relative_root.as_posix(),
                        cs.KEY_NAME: root.name,
                        cs.KEY_ABSOLUTE_PATH: root.resolve().as_posix(),
                    },
                )
                parent_identifier = self._get_parent_identifier(
                    parent_rel_path, parent_container_qn
                )
                self.ingestor.ensure_relationship_batch(
                    parent_identifier,
                    cs.RelationshipType.CONTAINS_FOLDER,
                    (cs.NodeLabel.FOLDER, cs.KEY_PATH, relative_root.as_posix()),
                )

        return eligible_files

    def process_generic_file(self, file_path: Path, file_name: str) -> None:
        relative_filepath = file_path.relative_to(self.repo_path).as_posix()
        relative_root = file_path.parent.relative_to(self.repo_path)

        parent_container_qn = self.structural_elements.get(relative_root)
        parent_identifier = self._get_parent_identifier(
            relative_root, parent_container_qn
        )

        self.ingestor.ensure_node_batch(
            cs.NodeLabel.FILE,
            {
                cs.KEY_PATH: relative_filepath,
                cs.KEY_NAME: file_name,
                cs.KEY_EXTENSION: file_path.suffix,
                cs.KEY_ABSOLUTE_PATH: file_path.resolve().as_posix(),
            },
        )

        self.ingestor.ensure_relationship_batch(
            parent_identifier,
            cs.RelationshipType.CONTAINS_FILE,
            (cs.NodeLabel.FILE, cs.KEY_PATH, relative_filepath),
        )
