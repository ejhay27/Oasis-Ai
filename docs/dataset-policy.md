# Dataset rights, provenance, and removal process

## Admission principles

Public accessibility is not a training license. Evaluate each dataset version and each source/license separately for machine-learning training, commercial use, redistribution, attribution, privacy, and derived-weight distribution. Do not rely on a dataset name or a webpage summary as proof of rights. This document is an engineering workflow, not legal advice; obtain qualified legal review for unresolved or commercial/distribution questions.

Permitted candidate sources may include properly licensed/public-domain footage (with status checked), suitable Creative Commons footage whose exact terms permit the intended use, OASIS-created footage, explicit contributor donations under a suitable agreement, or commercially licensed datasets whose terms cover training and intended distribution. No such source is pre-approved by this repository.

## Registry

`data/registry/datasets.json` is the canonical rights registry. Each `datasets[]` record includes:

- stable `dataset_id`, name, source/download location, version, acquisition date, and known rights holder;
- license name/URL and optional SHA-256 of the archived license text;
- separate `permissions.training`, `permissions.commercial_use`, and `permissions.redistribution` values (`allowed`, `prohibited`, `unknown`, or `restricted`);
- attribution requirement/text; privacy restrictions; known limitations;
- review name/date/evidence URI/notes; source-manifest checksum and count when known;
- status: `pending`, `approved`, `rejected`, or `takedown`.

The training gate requires `status: approved`, explicit `permissions.training: allowed`, matching dataset ID/version, and non-empty approval evidence. Commercial use and redistribution do not follow automatically from training permission. A pending or unknown status is blocked. The registry is editable project metadata, not an external legal certification.

The initial registry is intentionally empty. Never place guessed licensing claims in an `approved` record. Store a copy of license text/terms and source receipts in an approved access-controlled location; hash them in the registry where practical. Do not commit copyrighted footage or private contributor data in the source tree.

## Source and prepared manifests

A source JSONL row points to one source video and records its dataset ID/version, a useful item ID, a caption, and `caption_reviewed`. The preprocessing tool hashes the input bytes and records exact provenance on every clip. The caption is copied, not generated or verified by a model. Training requires a non-empty, manually reviewed caption.

Prepared rows record source SHA-256, source item ID/path, fixed frame paths and hashes, dimensions/FPS/clip interval, FFmpeg version/filter, and visual fingerprints. Audio is excluded. A preprocessing configuration and resulting manifest should be preserved; do not overwrite a manifest used by a run.

Frame path resolution is restricted to the prepared data root. Training revalidates registry eligibility, row fields, and processed-frame checksums before a run. Manifest and registry file SHA-256 values are stored in run metadata and checkpoint sidecars.

## Filtering and splits

- Probe/decode errors and truncated clips are failures, not silent skips.
- Inspect corrupt decodes, source resolution/aspect/FPS, clip lengths, duplicates, and sample quality before admitting data.
- Near-duplicate average-hash candidates are heuristics. Review false positives/negatives and retain the splitter sidecar report.
- Keep all clips from one exact source file in one split. Near-duplicate grouping is conservative. The split algorithm, seed, ratios, and input/output hashes are recorded beside the split manifest.
- Caption review is a human check; a caption can still be incomplete or inaccurate. Do not infer sensitive attributes or personal identity from a caption.
- Privacy/sensitive-content review is not automated by this code. Dataset owners must define proportionate screening, consent, retention, and access procedures before ingestion.
- Maintain a stable test set and ensure source/near-duplicate groups cannot cross train/validation/test.

## Provenance and takedown workflow

1. Log the complaint/request, requester, date, affected dataset/source IDs, and requested remedy without exposing unnecessary personal data.
2. Restrict the disputed item from new preprocessing/training while rights staff review it; change the registry record to `pending` or `takedown` when appropriate.
3. Verify source identity/checksums, license/contribution terms, and scope. Record reviewer, evidence, decision, and date.
4. Remove the item from future manifests/preprocessing and regenerate a new immutable manifest/split with recorded hashes. Do not mutate a manifest used by a past run.
5. Preserve an access-restricted audit trail for decisions, checksums, model versions, and recipients as appropriate.
6. Assess already-produced checkpoints with legal/privacy and model owners. Deleting a source file does **not** prove its influence is removed from existing weights. Do not promise unlearning; document whether retraining, a targeted removal study, distribution suspension, or no weight change is justified.
7. Recheck registry/model-catalog release status before publishing affected weights.

## Distribution gate

Training permission, commercial-use permission, and redistribution permission are distinct. Before distributing checkpoints or integrating them in a public Windows Studio build, review every dataset/license and any inherited model component for the intended use, attribution, restrictions, privacy, and takedown obligations. Keep model weights internal until that review is complete. The model catalog defaults OASIS training outputs to `distribution_status: not_cleared`.
