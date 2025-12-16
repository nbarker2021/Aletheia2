_**This is a new slice created to unify the CQE system with the phone data.**_

# CQE/MORSR v2 Slice

## Purpose and Scope

CQE/MORSR v2 is a complete, weight-agnostic overlay that fixes host behavior via a single adapter layer. It introduces several new features, including Ingress Parity Sync (IPS), Pose-as-Gauge (PaG), and Pre-Entry Pose Saturation (PEPS). The slice is ledger-first, Bell-honoring, and reversible-first, with no base weight changes.

## Architecture Overview

*   **MUA (Minimal Universal Adapter):** A single ingress/egress gate with Step-0 E₈ embedding on ingress and ledgering on every action.
*   **IPS (Ingress Parity Sync):** A one-tick, ingress-only GlobalParityMirror to canonicalize parity when hosts expose no hooks.
*   **PaG (Pose-as-Gauge):** 8-pose gauge transformations (global/substate/atom) with near-zero heat, used for alignment and edge validation.
*   **PEPS (Pre-Entry Pose Saturation):** Poses the incoming data through all 8 poses in the adapter staging area and deterministically selects the canonical representative before ingress.
*   **MORSR (Middle-Out Search with ΔΦ discipline):** Features escrow windows, plateau/half-life guards, policy channels (8 projectors), and a negative-proof cache.
*   **Ledger:** An append-only JSONL with receipts as micro-theorems (claim→evidence→cap→settlement).
*   **Verifier:** A deterministic replay and invariants checker.

## Global Invariants

The slice is governed by a set of global invariants, including:

*   **ΔΦ discipline:** Accepts require delta_phi ≤ 0 unless escrow is active.
*   **Reversibility-first:** Irreversible steps require a cap or heat debit at close.
*   **CBC (count-before-close):** Enumerate admissible ops or justify via cache hit.
*   **Policy channels:** Exactly eight, with spectra logged per pulse.
*   **Ethics potential:** E_ethics remains below a sustained threshold.
*   **Pose legality:** IPS is allowed once at pulse 0; PaG preserves admissible ops.
*   **Bell-honoring:** No cross-shard unseen state, no retrocausality, no unpriced erasure, and signatures on critical events.

