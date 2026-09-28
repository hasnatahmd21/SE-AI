"""SE Brain local Knowledge Fabric UI.

Run:
    python -m sebrain.ui --datasets ./datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from . import Config, SEBrain


APP_CSS = """
body,.gradio-container{background:#0a0a0f!important;color:#e8e8f0!important}
textarea{background:#05050a!important;color:#00ff88!important;font-family:monospace!important}
"""


def build_app(
    datasets_dir: str | Path = "./datasets",
    data_dir: str | Path = "./.sebrain_ui",
):
    config = Config(data_dir=Path(data_dir), log_level="WARNING")
    brain = SEBrain(config=config)
    brain.start()
    report = brain.connect_knowledge_fabric(datasets_dir)

    def analyze(task: str) -> str:
        if not (task or "").strip():
            return "Please enter a coding task."
        return brain.ask(task, top_k=5).to_english()

    def stats() -> str:
        current = brain.fabric_stats()
        coverage = current["coverage"]
        lines = [
            "SE BRAIN — KNOWLEDGE FABRIC",
            "",
            f"Total indexed records: {current['total_records']:,}",
            f"Datasets present: {coverage['dataset_count']}/58",
            f"Missing datasets: {', '.join(coverage['missing']) or 'None'}",
            "",
            "DATASETS",
            *[
                f"  {dataset_id}: {count:,}"
                for dataset_id, count in current["datasets"].items()
            ],
            "",
            "LANGUAGES",
            *[
                f"  {language}: {count:,}"
                for language, count in current["languages"].items()
            ],
            "",
            "SOURCE FILES",
            *[
                (
                    f"  {source['source_file']}: "
                    f"{source['inserted']:,} inserted, "
                    f"{source['duplicate_records']:,} duplicates, "
                    f"{source['conflicts']:,} conflicts, "
                    f"{source['errors']:,} errors"
                )
                for source in current["sources"]
            ],
        ]
        return "\n".join(lines)

    with gr.Blocks(title="SE Brain") as demo:
        gr.Markdown(
            "# ◈ SE BRAIN ◈\n"
            "Autonomous Software Engineering Brain · Knowledge Fabric"
        )

        with gr.Tab("ANALYZE"):
            task = gr.Textbox(
                label="Task Input",
                lines=6,
                placeholder="Build a FastAPI REST API with JWT authentication...",
            )
            with gr.Row():
                run = gr.Button("ANALYZE TASK", variant="primary")
                clear = gr.Button("CLEAR")
            output = gr.Textbox(
                label="Analysis Output",
                lines=32,
                buttons=["copy"],
            )
            run.click(analyze, task, output)
            clear.click(lambda: ("", ""), outputs=[task, output])

        with gr.Tab("DATASETS"):
            gr.Markdown(
                f"Loaded **{report['records']:,}** new records from "
                f"**{report['files']}** source files."
            )
            if report["missing_dataset_ids"]:
                gr.Markdown(
                    "⚠️ Missing expected datasets: "
                    + ", ".join(report["missing_dataset_ids"])
                )
            else:
                gr.Markdown("✅ D01–D58 dataset coverage is complete.")

            refresh = gr.Button("REFRESH STATS", variant="primary")
            stats_out = gr.Textbox(lines=34, show_copy_button=True)
            refresh.click(stats, outputs=stats_out)

    return demo, brain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="./datasets")
    parser.add_argument("--data-dir", default="./.sebrain_ui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo, _ = build_app(args.datasets, args.data_dir)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
