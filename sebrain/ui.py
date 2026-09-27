"""SE Brain local Knowledge Fabric UI.

Run from the package root:
    python -m sebrain.ui --datasets ./datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path
import gradio as gr

from .c01 import Config, SEBrainApp
from .c35 import BrainDatasetBridge
from .c34 import KnowledgeFabricLoader


def build_app(datasets_dir: str | Path = "./datasets", data_dir: str | Path = "./.sebrain_ui"):
    config = Config(data_dir=Path(data_dir), log_level="WARNING")
    brain = SEBrainApp(config=config)
    brain.start()
    loader = KnowledgeFabricLoader(brain.storage, datasets_dir)
    report = loader.load_all_datasets()
    bridge = BrainDatasetBridge(brain=brain, loader=loader)

    def analyze(task: str) -> str:
        if not (task or "").strip(): return "Please enter a coding task."
        return bridge.answer(task, top_k=5).to_english()

    def stats() -> str:
        s = loader.stats()
        lines = ["SE BRAIN — KNOWLEDGE FABRIC", "", f"Total records: {s['total_records']}", f"Datasets: {len(s['datasets'])}", ""]
        lines.append("DATASETS")
        lines.extend(f"  {k}: {v:,}" for k, v in s["datasets"].items())
        lines.append("")
        lines.append("LANGUAGES")
        lines.extend(f"  {k}: {v:,}" for k, v in s["languages"].items())
        return "\n".join(lines)

    css = """
    body,.gradio-container{background:#0a0a0f!important;color:#e8e8f0!important}
    textarea{background:#05050a!important;color:#00ff88!important;font-family:monospace!important}
    """
    with gr.Blocks(css=css, title="SE Brain") as demo:
        gr.Markdown("# ◈ SE BRAIN ◈\nAutonomous Software Engineering Brain · Knowledge Fabric")
        with gr.Tab("ANALYZE"):
            task = gr.Textbox(label="Task Input", lines=6, placeholder="Build a FastAPI REST API with JWT authentication...")
            with gr.Row():
                run = gr.Button("ANALYZE TASK", variant="primary")
                clear = gr.Button("CLEAR")
            output = gr.Textbox(label="Analysis Output", lines=32, show_copy_button=True)
            run.click(analyze, task, output)
            clear.click(lambda: ("", ""), outputs=[task, output])
        with gr.Tab("DATASETS"):
            gr.Markdown(f"Loaded **{report['records']:,}** records from **{report['files']}** JSONL files.")
            refresh = gr.Button("REFRESH STATS", variant="primary")
            stats_out = gr.Textbox(lines=30, show_copy_button=True)
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
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()
