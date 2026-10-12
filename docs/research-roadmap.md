# OASIS original-model research roadmap

**Status date:** 2026-10-12
**Program status:** Foundation work; no training dataset has been approved and no OASIS checkpoint has been produced.

This roadmap deliberately separates work that exists in this repository from work that remains planned. Passing software tests is not evidence of video-generation quality.

## Model categories and claim standard

- **External model:** weights and architecture originate outside OASIS. Record the actual creator, source, license, weight origin, and restrictions. Integration, hosting, download, or evaluation does not make the model OASIS-owned.
- **OASIS-adapted model:** derived from an external model by fine-tuning, adapters, distillation, or another permitted method. Record base model and license, training data and method, inherited/new weights, attribution, and redistribution terms. Never call it trained from scratch.
- **Original OASIS model:** an OASIS-controlled architecture and pipeline produced the learned weights in a documented training run. List any pretrained encoder, VAE, teacher, borrowed weights, and other inherited components. A new architecture without a completed training run is not a trained model.

“Fully trained from scratch” is allowed only when the complete weight and component provenance supports it. Training-data and model-weight rights remain separate review questions.

## Stages and gates

| Stage | Goal | Evidence required to exit | Current status |
|---|---|---|---|
| 1. Research and feasibility | Select a first tractable architecture, data rights process, compute strategy, and evaluation plan. | Architecture decision record; reviewed candidate dataset rights; measured hardware; small-run budget/cost estimate where relevant; release gates. | A targeted literature scan, software ADR, and local hardware snapshot are documented. Candidate dataset rights and a systematic benchmark/model-license comparison are not yet reviewed; no large-run cost can be quoted. |
| 2. Small experiment | Verify that an intentionally limited model can learn a short, low-resolution video distribution. | Actual real-data training log, reproducible checkpoint, finite train/validation metrics, fixed-seed samples, and human review. Synthetic tests do not meet this gate. | A temporary synthetic software smoke run passed the code path, but this research stage is **not started** on real, approved video. |
| 3. Data/training infrastructure | Make source rights, manifests, preprocessing, splits, configs, logs, resume, and errors traceable. | Unit tests plus a real small prepared dataset; immutable manifest/config hashes; verified resume; resource logs. | Eight unit tests and a synthetic end-to-end train/interrupt/resume/checkpoint/inference/evaluation smoke test passed. Actual FFmpeg ingestion and real-data operation remain untested. |
| 4. Controlled experiments | Compare justified architecture/objective/data/resolution/batch variants. | Predeclared experiment matrix, repeated runs/seeds where practical, actual resource and validation records, failed-run notes. | Planned. No results exist. |
| 5. First original OASIS model | Produce the first verifiable checkpoint from the OASIS training code. | Registered checkpoint with model card, run/config/dataset hashes, provenance, test evaluation, and inference sample. | Not achieved. |
| 6. Model improvement | Improve temporal continuity, prompt adherence, visual quality, and efficiency based on measurement. | Controlled comparisons on stable splits plus human assessment; no cherry-picked claims. | Planned after Stage 5. |
| 7. Desktop integration | Integrate verified compatible weights without coupling Studio startup to research training. | Windows compatibility tests, safe checkpoint import, model version/capability UI, export tests, explicit research actions. | No desktop application source exists in this checkout; backend adapter boundary only. |

## Stage 2 hypothesis and limits

ADR 0001 chooses a tiny pixel-space conditional diffusion experiment: 8 frames, 16×16 RGB, 4 fps preprocessing defaults, small 3D convolutions, byte-level trainable text embedding, 100 DDPM noise levels, and no pretrained components. The goal is to validate the end-to-end training/evaluation path, not to establish useful prompt adherence or compete with any external system.

If a dataset with documented training permission and suitable captions is unavailable, use synthetic data only for software/debugging tests and label it as such. Synthetic fixture metrics are not a model-quality result.

## Scale-up gates

Do not scale until all of the following are true:

1. Rights-approved, provenance-complete data exists and has a stable train/validation/test split.
2. Multiple tiny runs complete without corrupting checkpoints; interrupted/resumed runs reproduce expected state on the same software/hardware setup.
3. Validation measurements are finite and recorded from real data; generated sample quality has been independently reviewed.
4. The run has measured throughput and peak memory on target hardware. Any cost projection states GPU-hours, provider rate source/date, storage/transfer, total, and uncertainty.
5. The next experiment changes a documented variable and has predeclared acceptance criteria.
6. Legal review approves the intended weight/model distribution path separately from training permission.

No cloud provider is configured. Never launch remote training or incur remote compute charges without explicit user approval.
