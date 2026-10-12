# Evaluation protocol

A completed optimization loop is not evidence that a video model is useful. Evaluation must use a stable held-out test split and preserve exact model/checkpoint, dataset/split manifest, registry, seed, preprocessing, sample count, software, and hardware references.

## What exists

`oasis-research evaluate` currently:

1. verifies the OASIS checkpoint hash/sidecar and loads the fixed architecture from safetensors;
2. validates the rights-approved prepared manifest and selects its `test` split only;
3. records held-out epsilon-prediction MSE for a fixed, seeded sample of test clips;
4. generates a fixed-seed sample for each selected held-out caption and saves numbered PNG frames;
5. records basic adjacent-frame pixel differences and exact adjacent duplicate-frame rate as diagnostics, not quality scores;
6. writes `evaluation.json` plus a blank `human-review.csv` with prompt, visual quality, temporal coherence, and prompt-adherence rating fields.

The held-out epsilon MSE is a diffusion-objective diagnostic. It is not a perceptual/video-quality score. Pixel differences can reflect valid or invalid motion; they do not measure object identity, semantic continuity, or prompt adherence. The code does not compute FVD or pretrained text-video similarity; no metric is fabricated.

## Human review

Use diverse, predeclared prompts and review generated clips without cherry-picking. At minimum, review:

- requested objects/actions/scenes and spatial relationships;
- object integrity and identity persistence;
- motion coherence and direction;
- flicker, jumps, duplicated frames, and appearance drift;
- composition, artifacts, color stability, diversity;
- harmful, private, or unexpected outputs relevant to intended deployment.

Have more than one reviewer where possible, record disagreements and sampling conditions, and keep failed/poor samples. A PNG frame sequence must be watched as a sequence (or assembled by a documented, reproducible renderer) before drawing temporal-quality conclusions.

## Metrics to add only with a validated implementation

Potential future metrics include FVD when the encoder, implementation, evaluation split, sample count, preprocessing, and rights are all documented; suitable text-video similarity; spatial image metrics where the task supports them; and validated temporal consistency measures. Metrics that use pretrained encoders introduce their own creator/license/weight provenance and must be listed in the model/evaluation report. No one scalar captures video quality.

For each metric, report implementation/version, evaluation set and checksum, preprocessing, sample count, random seeds, comparison conditions, confidence/uncertainty, known biases, and whether the same split was used to tune the model. Keep a stable, hidden test split where feasible.

## Baselines and release gates

Do not claim improvement over an external model unless comparison rights permit it and both systems use equivalent prompts, output constraints, sample counts, seeds where possible, and review conditions. Record the external model's actual identity/license. Keep blind human comparisons and avoid cherry-picked grids.

Every major experiment should predeclare measurable criteria. Possible states are experimental, research preview, internal alpha, limited public release, and production. Production requires a representative stable test set, reproducible results, human quality/safety review, deployment compatibility, documented failure modes, security review, and cleared dataset/model distribution rights.
