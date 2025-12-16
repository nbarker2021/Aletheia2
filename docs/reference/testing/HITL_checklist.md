# SPR001 — Human-in-the-loop Checklist

## H1 Triad Adequacy (TAC)
- [ ] Review triad+inverse examples (sample 10) for coverage & exclusivity.
- [ ] Confirm Underverse 8×8 stance for any TAC-fail cases.

## H2 DetectorGate Curve
- [ ] Inspect learned monotone θ grid (heatmap + CSV).
- [ ] Compare false-accept/false-hold deltas vs fixed θ=0.60.

## H3 AcceptanceGate Policy
- [ ] Review Pareto frontier + top-decile band; check leakage vs acceptance uplift.

## H4 Assembly/DNA
- [ ] Validate C[8] candidates stitching; check barycentric mix and DNA replay hash.

## H5 TeleTrail
- [ ] Open teletrail_release.jsonl for each tick; confirm event coverage and parity hashes.

## H6 MRS Replay
- [ ] Rebuild from mrs.json5; confirm bit parity for indices/e8_proj and semantic parity overall.
