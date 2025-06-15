# Two-Decoder Pipeline

This experimental pipeline uses a small decoder-only VLM to compress multimodal context. The hidden states are refined through a lightweight **Linguistic Refiner** module before being consumed by a larger solver decoder that generates the final answers.

```
"<image> text to compress" -> Decoder-only VLM -> Linguistic Refiner -> Solver Decoder
```

`LinguisticRefiner` implements 2-4 Transformer layers with full attention and adaptive gating as proposed in the [Linguistic Token Refiner](https://arxiv.org/pdf/2406.11831) paper.

An example model wrapper is provided in `llava.model.two_decoder_pipeline.TwoDecoderPipeline`.

### Dataset and Loader

`llava.dataset.compression_dataset` includes `VLMCompressionDataset` and `build_compression_dataloader` helpers for preparing image–text pairs when using the pipeline with models such as Qwen2.5-VL.
