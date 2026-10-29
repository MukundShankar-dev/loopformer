"""The same small queued streaming API for Hugging Face and local serving."""

from typing import Callable

import gradio as gr


def create_app(predict: Callable, *, model_url: str | None = None) -> gr.Blocks:
    with gr.Blocks(title="LoopFormer · live inference", analytics_enabled=False) as app:
        gr.Markdown("# LoopFormer\nFollow a graph one recurrent loop at a time. "
                    "This endpoint streams real model predictions; it does not substitute reference answers.")
        links = "[Open the interactive playground](https://mukundshankar-dev.github.io/projects/loopformer/demo/)"
        if model_url:
            links += f" · [Download the model]({model_url})"
        gr.Markdown(links)
        request_input = gr.JSON(label="Graph request: mapping, start, steps")
        result = gr.JSON(label="Streamed trace")
        run = gr.Button("Run model", variant="primary")
        run.click(predict, inputs=request_input, outputs=result, api_name="predict",
                  concurrency_limit=1, stream_every=0.05)
    return app.queue(max_size=16, default_concurrency_limit=1)
