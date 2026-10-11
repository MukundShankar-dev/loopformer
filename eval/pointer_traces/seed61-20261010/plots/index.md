# Ordinary trace baseline comparisons

Separate addition to the completed population figures. Trace termination is emitted-state length plus EOS, not latent loop count.

- [01_quality_by_depth](./01_quality_by_depth.png) · [PDF](./01_quality_by_depth.pdf)
- [02_trace_failures](./02_trace_failures.png) · [PDF](./02_trace_failures.pdf)
- [03_graph_strata](./03_graph_strata.png) · [PDF](./03_graph_strata.pdf)
- [04_sft_training](./04_sft_training.png) · [PDF](./04_sft_training.pdf)

Ordinary SFT adapts all Qwen weights on the same examples as R, using full-vocabulary autoregressive CE. It has more trainable capacity than R. The final controller additionally saw counts 1–63; this is not a fully exposure-matched stopping comparison.
