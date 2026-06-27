"""Input/Output: parsing, normalization, preprocessing, and artifact writing."""

from .parser import load_file, normalize, to_records, detect_format
from .preprocess import Preprocessor, preprocess_file
from .writer import (
    ensure_run_dir, write_normalized, write_profile, write_run_plan,
    write_method_output, write_consensus_edges, write_consensus_clusters,
    write_cluster_members, write_artifact_manifest,
)

__all__ = [
    "load_file", "normalize", "to_records", "detect_format",
    "Preprocessor", "preprocess_file",
    "ensure_run_dir", "write_normalized", "write_profile", "write_run_plan",
    "write_method_output", "write_consensus_edges", "write_consensus_clusters",
    "write_cluster_members", "write_artifact_manifest",
]
