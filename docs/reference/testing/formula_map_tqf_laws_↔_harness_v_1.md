# Formula Map — TQF Laws ↔ Harness (v1.3)

**Legend:** Green = solid; Yellow = partial; Red = placeholder.

| ID | Law / Capability | Equation (informal) | Harness Location | Meaning | Test Status | Next Action |
|---:|---|---|---|---|---|---|
| 1 | QME | \(\mathcal{Q}(a,b)=(a\oplus b, a\odot b, a\ominus b, a\oslash b)\) | `tqf/core/qme.py:qme_tuple` | Universal fingerprint | **Green** | Property tests for involution/closure |
| 2 | Derivative Balance | \(n=(a,b)+(c,d), \max≤\tfrac{1}{2}\text{base}\) | `tqf/core/reduce.py:reduce_derivative` | Aperture normalizer | **Green** | Vectorize; SL5 idempotence proofs |
| 3 | Entropy Gate | ΔS gate | `tqf/core/entropy.py:entropy_update` | Operational entropy | **Green** | Persist Ramanujan trail |
| 4 | CNF Path‑Independence | CNF(A→B)=CNF(A→C→B) | `tqf/core/embed.py:cnf_decode` (3×E8 glue) | Canonical lattice NF | **Yellow** | Implement per‑block E8 + glue + φ receipts |
| 5 | CRT Defect | gcd>1 ⇒ defect receipt | `tqf/core/crt.py:crt_fingerprint` | Aliasing audit | **Yellow** | Add Bézout witness/remedy |
| 6 | Octet Commute | f(B₁∪B₂)=f(B₁)∪f(B₂) | `tqf/core/buckets.py:bucket_map` | Batch = atomic | **Yellow** | Wire canonical parity‑triple |
| 7 | Φ‑Probe | φ·ΔS vs φ²·ΔS | `tqf/phi_probe.py:phi_probe` | Deterministic tiebreak | **Green** | N‑way forks, persist φ tuple |
| 8 | Taxicab/Cabtaxi | multi cube forms | `tqf/core/taxicab.py` | Phase gates | **Yellow** | Emit witnesses & ΔS expectations |
| 9 | Scalar Families | {add,mul,fact,cum}_B | `tqf/core/scalars.py` | Base grammar | **Yellow** | Finish factorial/cumulative |
| 10 | HP/SP Conversions | HPx ↔ HPy (lossless) | `tqf/core/hp.py` | Universe conversions | **Yellow** | Wire to CNF & scalars |
| 11 | **Real CNF (E8/Leech)** | nearest‑vector + legal glues | `embed.py:cnf_decode` + `autos.py` | Boundary‑only receipts, Co₀ certs | **Yellow** | Implement decode & certificate; add ε policy |
| 12 | **ALENA tri‑lattice** | triadic energy H_𝒜; ΔS on boundaries | `alena.py:{step,project_out,context_shift}` | Contextual shift across 3 lattices | **Yellow** | Implement ops; emit ALENA receipts |
| 13 | **Q‑Mass & Horizon** | M_core, M_halo, M_total | `receipts.py` (+ qmass calc) | Idea weight & boundary audit | **Yellow** | Add fields; unit tests for invariance |

## Unit test plan (sketch)
- `test_cnf_path_independence()` — braids + certificates
- `test_boundary_only_emission()` — interior vs boundary walks
- `test_phi_probe_determinism()` — synthetic degeneracy
- `test_crt_defect_with_bezout()` — witness & remedy
- `test_bucket_commutation()` — property‑based
- `test_scalars_crosswalk()` — B8↔B16↔B32
- `test_hp_roundtrip_lossless()` — receipts equality
- `test_qmass_invariance_sl7()` — M_core stable
- `test_alena_no_env_closure()` — ΣΔS=0 cycles
- `test_bleedover_audit()` — balanced horizon ΔS

