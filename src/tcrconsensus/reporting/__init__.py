"""Report generation: JSON, Markdown, figures."""

from .report import (
    generate_report, write_json_report, write_markdown_report, generate_figures,
)

__all__ = [
    "generate_report", "write_json_report", "write_markdown_report", "generate_figures",
]
