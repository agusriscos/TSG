# N=1500 / SNR=20 dB bundle

Separate output tree for the joint operating point **N=1500**, **SNR=20 dB**.

Generated (UTC): `2026-11-24T11:32:33Z`

## Layout

- `meta/operating_point.json` — fixed OP metadata from the B3 difficulty curve
- `reports/phase2_hard_frontiers/` — B3 panel + difficulty slice for this cell
- `reports/phase2_information/` — B1 on the same SNR=20 records
- `reports/phase2_phase_maps/` — B5 single-cell grid (or STATUS if skipped)
- `reports/phase2_surrogates/` — STATUS (no SNR=20 hook yet)
- `reports/phase2_invariance/` — STATUS (no SNR=20 hook yet)
- `paper_bundle/` — flat PNG + CSV copies with `b*_` prefixes

## Notes

- Difficulty-cell ensembles use R=20 (not the clean-panel R=30).
- This folder is independent of `outputs/paper_bundle/` (headline there is N=500 / SNR=20).
