# ADR 0001 — first experimental OASIS video architecture

- **Status:** Accepted for a deliberately small research prototype; not a production architecture.
- **Date:** 2026-10-12.
- **Architecture ID:** `oasis.tiny_video_diffusion.v1`.
- **Model category:** Intended original OASIS architecture. It does not constitute a trained model until a checkpoint is created by an actual, recorded run.

## Decision

Implement a compact **pixel-space, text-conditioned DDPM** for fixed-shape RGB clips. The research config defaults to `T=8`, `H=W=16`, 3 channels, a small 3D-convolutional encoder/middle/decoder, a trainable UTF-8-byte embedding, sinusoidal diffusion-time conditioning, and epsilon prediction with a linear beta schedule.

The model runs at low resolution and does one spatial downsample; time is not downsampled. The text representation is a simple learned byte embedding with a fixed byte limit; there is no external tokenizer or pretrained text encoder. Inference uses basic ancestral DDPM sampling. The first release would be a research experiment only. For the default `base_channels=32`, `condition_dim=128` architecture, the current module shapes imply **604,131 trainable parameters**; every run also logs the instantiated parameter count. This small parameter-state figure excludes activation/temporary/runtime memory and is not a training-memory estimate.

## Why this direction

- It exercises temporal modeling, text conditioning, diffusion training, checkpointing, and sampling without first depending on a pretrained video autoencoder or third-party encoder.
- The entire model can be initialized randomly and audited in this repository; the current architecture has no teacher, borrowed weights, external encoder, or VAE.
- The small fixed shape makes data-pipeline and CPU/unit-test debugging possible. It is still not expected to train usefully on an ordinary laptop or tiny, low-diversity dataset.
- The byte encoder keeps tokenization provenance simple, at the cost of weak language modeling and a small fixed context.

## Components and training objective

| Component | Current implementation |
|---|---|
| Input | RGB pixel clip tensor `[B,C,T,H,W]`, scaled to `[-1,1]`; fixed shape per config. |
| Spatial modeling | 3D convolutions with one spatial downsample/upsample and residual blocks. |
| Temporal modeling | 3D convolution kernels span adjacent frames; no temporal downsampling or long-range attention. |
| Text conditioning | Trainable embedding of UTF-8 bytes (IDs 1–256, 0 padding), mean pooled and combined with sinusoidal diffusion-time embedding. |
| Image conditioning | Not supported. Text-to-video does not imply image-to-video capability. |
| Latent representation | None; diffusion runs in pixel space. |
| Objective | Predict Gaussian noise added according to a linear DDPM beta schedule using MSE. |
| Sampling | Ancestral DDPM, one network evaluation per configured diffusion timestep. No classifier-free guidance is implemented. |
| Pretrained/inherited components | None in the code path. Any later introduction requires an ADR update and explicit origin/license/provenance records. |
| Weights | Random initialization only; learned weights exist only after a recorded OASIS training run. |

## Known limitations and risks

- 16×16 frames are far below useful visual detail. The model may learn only color/texture statistics; no useful generation is promised.
- A byte-level mean embedding is not a capable language encoder. Word order and compositional semantics are poorly represented.
- Fixed short clips and local 3D convolutions limit long-term motion, object persistence, and camera control.
- Pixel-space diffusion is inefficient at useful resolution. The baseline sampler performs 100 sequential denoiser calls by default.
- A small dataset can overfit and provide no meaningful test coverage; captions can be noisy even when manually checked.
- Pixel-difference diagnostics do not establish semantic temporal coherence. Human review and suitable independent metrics are required.
- This direction is an engineering feasibility choice, not a claim of novelty over published research. A literature review and competitive architecture comparison remain Stage 1 work.

## Acceptance criteria before considering a successor

1. Complete an actual rights-approved short run and preserve config, manifest/registry hashes, framework, hardware, random state, metrics, checkpoints, and failures.
2. Demonstrate deterministic fixed-seed sampling and verified resume on the same environment.
3. Evaluate the held-out test split and use a blinded human-review protocol. Do not report “quality” from training loss alone.
4. Measure inference/training memory and throughput on the actual deployment/training hardware.
5. Compare alternatives only after defining fair prompts/splits/seeds and obtaining any baseline model/data rights.
6. Revisit a latent video autoencoder or attention-based model only with a documented licensing plan for every inherited component and a measured compute budget.
