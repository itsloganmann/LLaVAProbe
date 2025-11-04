"""Entry point shim that delegates to the analysis pipeline."""

from __future__ import annotations

from analysis.pipeline_runner import main as pipeline_main


def main() -> None:
    """Execute the modern analysis pipeline."""

    pipeline_main()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
