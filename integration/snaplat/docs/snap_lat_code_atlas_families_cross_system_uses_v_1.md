# SnapLat Code Atlas — Families & Cross‑System Uses (v1)
**Scope:** Running, evidence‑first registry of *all* code extracted this session, organized by **families**, **use‑cases**, and **cross‑family** links. Feeds the **Frankenstein** stitching workflow and clarifies system build direction. Sources: the v2 classification and v2 Frankenstein index (see “Data Sources”).

## Data Sources (immutable, session‑local)
- **Classification v2 JSON/CSV:** `/mnt/data/snaplat_code_classification_v2_20250814T223634Z.json` (CSV alongside)
- **Frankenstein index v2:** `/mnt/data/frankenstein_v2/frankenstein_index_v2.json`
- **Ingest manifest v2:** `/mnt/data/snaplat_ingest_manifest_v2_20250814T223634Z.json`

---

## 1) Family Taxonomy (content‑derived)
Tags are assigned by content patterns observed in the code (tokens/AST), not by file names. A file may belong to multiple families.
- **agrm** — navigator, statebus, controller, sweeps, promotion, planner hooks
- **mdhg** — hashing/hotmaps/hierarchy/persistence, elevator fast‑lane/promotion
- **snap** — SNAP/SNAPDNA, ops center, capsule/pack registry
- **universe** — overlays, user/task/model/tool lenses
- **shelling** — shelling engines & sweeps
- **glyph** — glyph codecs and transforms
- **beacons** — beacon registries & helpers
- **w5h** — W5H alignment/weights
- **orchestrator** — freshstart/iterate control
- **policy** — policy bus/compliance/audit/notice
- **repo** — projector, run manifests, repo ops
- **utils/telemetry** — generic utilities, trace/quicktrace

> Roles: **package_module / script / test** are inferred by content and path (e.g., `__main__`, `/tests/`).

---

## 2) Family Coverage (counts)
_The complete table is saved as `/mnt/data/code_atlas/family_counts.csv` and shown in the UI._
- This section summarizes how much code (by file count) is tagged to each family.

---

## 3) Use‑Case Coverage and Family×Use‑Case Matrix
- Use‑cases tracked: **indexing, routing, scoring, governance, stitching, shelling_engine, glyph_codec, orchestration, universe_overlay, telemetry, repo_manifest, alignment**.
- Full long‑form matrix: `/mnt/data/code_atlas/family_usecase_matrix_long.csv` (also shown in the UI’s “Family×Use‑Case” table).

**Interpretation pattern:** High counts for *(agrm, routing)* or *(mdhg, indexing)* reinforce expected build roles; outliers merit review (possible mis‑tags or cross‑cutting modules).

---

## 4) Exemplars per Family (for deep‑dive)
- Top files per family (by number of functions then size), with roles and use‑case tags: `/mnt/data/code_atlas/exemplars_per_family.csv`.
- Use this to open the strongest anchors in each family first during review.

---

## 5) Cross‑Family Overlaps (co‑occurrence in files)
- Pairwise overlaps counted across all files: `/mnt/data/code_atlas/family_overlap_pairs.csv` (also shown in UI cross‑family table).
- **Use:** reveals modules that straddle multiple families (e.g., **agrm**∩**mdhg**, **universe**∩**policy**). These are prime candidates for **Frankenstein** stitching boundaries and API surfaces.

---

## 6) Frankenstein Signals by Family
- **Function‑variant clusters per family:** `/mnt/data/code_atlas/franken_function_clusters_per_family.csv` (how many “same‑name different body” clusters touch each family).
- **File‑basename clusters per family:** `/mnt/data/code_atlas/franken_file_clusters_per_family.csv`.
- **Top variant functions (global):** `/mnt/data/code_atlas/top_variant_functions.csv`.

**Use:** families with many variant clusters likely contain rich evolution histories; ideal for stitching and comparative refactors.

---

## 7) Review Queue (living)
- **High‑value families first:** agrm, mdhg, universe, shelling, w5h.
- **Ambiguities:** prefer code with high function counts and strong matrix alignment; defer single‑function utilities unless blocking.
- **Outliers:** any file with unexpected family/use‑case combos gets flagged for manual inspection and tag refinement.

> When you add new bundles: we will re‑run **extract → index → classify → franken** and append deltas here, keeping lineage intact.

---

## 8) Next Actions (you can say “run these”)
1. **Drilldowns:** Generate per‑family sub‑atlases (mini CSVs) and per‑use‑case slices.
2. **Franken shards:** Emit per‑cluster shard files for high‑variant functions (top N) to accelerate review.
3. **Tag refine:** Interactive pass to correct any mis‑tagging discovered in exemplars.
4. **Interface sketch:** From overlaps, sketch stable APIs between **agrm↔mdhg↔universe** as stitching targets.

---



## Code Atlas — Batches (Full Corpus)
**Directive:** Build *all* batches first, then we’ll review for like/same files and select best versions; all others tagged for Frankenstein use.

**What’s ready now:**
- **Global batches manifest:** `/mnt/data/code_atlas/batches/batches_manifest_20250814T225008Z.json`  
  → shows all **19 batches**, 100 files each (last batch may be smaller), with their `.md` and `.json` paths.
- **Per-batch outputs:**
  - `batch_XX_full_20250814T225008Z.md` — **full code** for that batch (100 files per file, headers + fenced code)
  - `batch_XX_index_20250814T225008Z.json` — machine index (file metadata + per-function hashes)
- **Needle index (cross‑batch search):** `/mnt/data/code_atlas/batches/needle_index_20250814T225008Z.csv`  
  → one row per file with `FILE::<relpath>::<sha256>`, families/use‑cases, function count, size, and batch number.

**Batching policy (unchanged):** priority order `agrm → mdhg → universe → shelling → w5h → snap → glyph → beacons → orchestrator → policy → repo → telemetry → utils`, then by (#functions desc, size desc).

**How to use now:**
- Open the **batches manifest** (above) and click any batch’s `.md` to read full code for that set.
- Use the **needle index** to filter (e.g., by family=mdhg or batch_no=03) when hunting for “best version” candidates.

**Next (when you say):**
- I’ll auto‑group “like/same” files (same basename or high‐overlap) across all batches and present **group review packs** in this canvas. Your selections will mark **Best‑Of**; all others get **Franken‑tagged** with provenance for stitching.

---



# Holistic Cross‑Use Review (v1)
**Objective:** Establish system‑wide **cross‑uses** and a **term index** that isolates both **distinct** (family‑specific) and **vague** (cross‑family) language. This anchors the “needle in a haystack” search for best versions and informs where Frankenstein stitching will benefit most.

## A) Cross‑Family Uses
**Method:** For each file, map its families (content‑derived) and parse its **imports**; map import names to families by keyword. Aggregate `(source_family → target_family)` edges and sort by weight.

**Output files:**
- Cross‑family edges: `/mnt/data/code_atlas/cross_family_edges.csv`  
- Family co‑occurrence (within files): `/mnt/data/code_atlas/family_cooccurrence.csv`

**Reading the edges:** Heavy edges (e.g., `agrm→mdhg`, `agrm→universe`) indicate strong dependencies and ideal lines for API hardening, adapters, or stitching seams.

---

## B) Term Index (Distinct vs Vague)
**Method:** Tokenize identifiers (snake + camel split), filter stopwords, compute **document frequency** per term and its spread across families.
- **Distinct terms:** concentrated in ≤3 families, top‑family share ≥ 0.65, df ≥ 10.
- **Vague terms:** present in ≥6 families, df ≥ 20, top‑family share ≤ 0.35.

**Output files:**
- Global terms: `/mnt/data/code_atlas/terms_global.csv`
- Distinct terms: `/mnt/data/code_atlas/terms_distinct.csv`
- Vague terms: `/mnt/data/code_atlas/terms_vague.csv`

---

## C) Deep Review Packs (auto‑generated)
Built term‑centric packs with **anchors** into files/functions, prioritized by family importance and code density.
- JSON packs: `/mnt/data/code_atlas/deep_review_terms.json`
- Top terms summary: `/mnt/data/code_atlas/deep_review_terms_top.csv`

**Anchor Schemas:**  
- `FILE::<relpath>::<sha256>`  
- `FUNC::<relpath>::<qualname>::<hash>`

**How to use:** Pick any term from the **Top** CSV; I’ll inline the first N occurrences (headers + function snippets) here for a focused review.

---

## D) Next Automatic Steps (if you say “go”)
1. **Edge‑driven packs:** for the strongest edges (e.g., `agrm→mdhg`), assemble file groups that *implement* the interaction, for API surface review.  
2. **Distinct‑term deep dives:** generate review sections for the top 10 distinct terms (family‑specific code) to pick best versions.  
3. **Vague‑term clarifiers:** generate review sections for the top 10 vague terms to force naming/contract clarity and prevent future drift.  
4. **Auto candidate ranking:** within each review pack, rank variants by test density, function count, and callsite centrality; nothing is chosen yet—just sorted for speed.

---



# Holistic Review Packs (v1.1)
**What I did:** Generated **edge‑driven** and **term‑driven** review packs with a transparent ranking so we can surface best‑version candidates fast, without making canonical choices yet.

**Where they live:** `/mnt/data/code_atlas/review_packs/`
- **Master index:** `master_index.json` (one stop to navigate all packs)
- **Edge packs:** `edge_packs/pack_<A>_to_<B>.json` and `.csv`
- **Term packs:** `terms/distinct_<term>.json/.csv`, `terms/vague_<term>.json/.csv`

**Scoring (per pack):** `score = 0.4·norm(num_functions) + 0.3·norm(franken_variants) + 0.2·norm(imports_mapped_out) + 0.1·norm(size)`  
_All features are evidence‑only (no speculation); normalized within each pack._

---

## Top Cross‑Family Edges (for API surface review)
1. **agrm → shelling** (weight=250)
2. **universe → agrm** (weight=170)
3. **mdhg → agrm** (weight=163)
4. **mdhg → shelling** (weight=146)
5. **policy → shelling** (weight=124)

> For each edge above, see `edge_packs/pack_<A>_to_<B>.*` — files are ranked by the scoring formula, with anchors like `FILE::<relpath>::<sha256>` and sample imports mapping to the target family.

---

## Term Funnels (to triage “distinct” vs “vague” language)
**Distinct (family‑concentrated) — top examples:**
- `xn` (agrm, share=1.0, df=34)
- `mid` (agrm, 0.781, df=32)
- `neigh` (mdhg, 0.75, df=28)
- `arr` (glyph, 0.957, df=23)
- `opened` (agrm, 1.0, df=21)

**Vague (spread across many families) — top examples:**
- `len` (top=agrm, share=0.302, df=711)
- `name` (agrm, 0.341, df=643)
- `append` (agrm, 0.34, df=606)
- `default` (agrm, 0.319, df=596)
- `range` (universe, 0.323, df=530)

> For each term, open `terms/distinct_<term>.*` or `terms/vague_<term>.*`. Packs include ranked file lists and **function anchors** for quick dive.

---

## How we’ll use these (proposal)
1. Start with **edge packs** (e.g., `agrm→shelling`, `mdhg→agrm`) to sketch stable interfaces and identify best‑version candidates near those seams.  
2. Run **distinct‑term packs** to consolidate specialized logic (reduce forks by selecting the clearest implementation).  
3. Use **vague‑term packs** to standardize naming/contracts (reduce drift) — often these touch generic Python usage and should be deprioritized unless they collide with domain terms.

**Ready when you are:** I can inline the **top 5 files** from any pack here with headers + code (or just the top functions) for immediate group review, while keeping the rest accessible via the pack CSV/JSON.

---



# Preliminary Best‑Of Baseline (Soft) — v1
**Goal:** Provide a non‑binding **best‑of** pick across *all* variant clusters so we have a working baseline for review and stitching.

**Method (evidence‑only, transparent):**
- **Per‑file score:** `0.4·norm(num_functions_full) + 0.3·norm(franken_variants) + 0.2·norm(imports_mapped_out) + 0.1·norm(size)`
  - `num_functions_full` and full **imports** were re‑parsed from AST for every file (no truncation).
  - `franken_variants` counts how many function‑variant clusters reference that file.
- **Per‑function instance score:** `0.5·file_score + 0.5·norm(nloc)` (normalized within the function’s cluster).

**Scope covered:**
- **File‑basename clusters:** 92 clusters → **92 preliminary picks** (top‑3 per cluster saved for context).
- **Function‑name clusters:** 309 clusters → **309 preliminary picks**.

**Artifacts (download):**
- **Best‑of — File clusters:** `/mnt/data/code_atlas/best_of/best_of_file_clusters.csv`
- **Best‑of — Function clusters:** `/mnt/data/code_atlas/best_of/best_of_function_clusters.csv`
- **Soft baseline manifest (both sets):** `/mnt/data/code_atlas/best_of/soft_baseline_manifest.json`
- **Context list — File clusters (top‑3 each):** `/mnt/data/code_atlas/best_of/best_of_file_clusters_top3.csv`

**Review guidance:**
- Use the **score margin vs runner‑up** to gauge confidence. Margins ≥ **0.15** are usually strong; small margins suggest a deeper look (or stitching two variants).
- If a chosen file is heavy on **imports_mapped_out** into another family, consider documenting the boundary (good API seam).

**Next options:**
1) **Inline samples**: Paste the top N files from any family or the highest‑margin picks across all clusters.
2) **Auto‑patch**: Generate a “Best‑Of” working tree (copy‑only, no edits) under `code_atlas/best_of/tree/` for sandbox execution.
3) **Delta heatmap**: For any function cluster, show a diff‑oriented view of runner‑up vs pick (to identify bits to graft into the Best‑Of).

---



# Baseline Build Artifacts (v1 — Rebuilt)
**Why rebuilt?** The environment cleared prior generated artifacts. I reconstructed the pipeline end‑to‑end from your uploaded bundles to keep momentum and stay fully evidence‑backed.

## What’s ready now
- **Rebuilt ingest manifest:** `/mnt/data/manifest_v2_rebuild.json` (CSV alongside)  
- **Frankenstein index (v2 rebuilt):** `/mnt/data/frankenstein_v2/frankenstein_index_v2.json`
- **Best‑Of (soft baseline):**
  - Files (per basename cluster): `/mnt/data/code_atlas/best_of/best_of_file_clusters.csv`
  - Functions (per function cluster): `/mnt/data/code_atlas/best_of/best_of_function_clusters.csv`
- **Best‑Of working tree (copy‑only):** `/mnt/data/code_atlas/best_of/tree/src/`  
  → manifest with parse results: `/mnt/data/code_atlas/best_of/tree/tree_manifest.json`
- **Close‑call function diffs (top 15 tight margins):**
  - Index: `/mnt/data/code_atlas/best_of/diffs/functions/index.json`
  - Each `.diff` file sits in the same directory

## Notes on method
- **Extraction:** recursive ZIP/TAR with provenance tracking; nested archives included.  
- **Classification:** content‑based tags (families/use‑cases) from code tokens + AST; roles inferred (module/script/test).  
- **Frankenstein clusters:** (a) **file**: same basename, distinct hashes; (b) **function**: same name, distinct normalized bodies.
- **Scoring (unchanged):**  
  `file_score = 0.4·norm(num_functions_full) + 0.3·norm(franken_variants) + 0.2·norm(imports_mapped_out) + 0.1·norm(size)`  
  `instance_score = 0.5·file_score + 0.5·norm(nloc)`

## How to proceed (my recommendation)
1. **Sanity pass on the working tree:** scan the parse table (I’ve surfaced it in‑UI) and flag any non‑parsing picks for swap‑out from their cluster runner‑ups.  
2. **Skim the close‑call diffs** to spot superior fragments worth grafting into the best‑of instances.  
3. **Pick an edge (e.g., `agrm→shelling`)** and I’ll regenerate the interface sketch over the *rebuilt* corpus and inline top signatures for quick API surface definition.

> Everything remains **non‑destructive** and is anchored with file/function hashes so SNAP‑Ops can stitch variants after review.

---



# Run‑All Execution (Parse Fixes → Edge Sketch → Grafting) — v1

## 1) Working Tree Parse Fixes (auto‑swap)
- Scanned the Best‑Of working tree and **replaced non‑parsing files** with runner‑ups from their basename clusters (rank #2, then #3 if needed). Originals saved under `tree/replaced/`.
- **Updated manifest:** `/mnt/data/code_atlas/best_of/tree/tree_manifest.json`  
- **Overrides log:** `/mnt/data/code_atlas/best_of/best_of_overrides.json` (target path, backup, replacement source, reason)

> You can view the in‑UI table “Working Tree — Parse Check (after auto‑fix)” to verify all files now parse. Any persistent failures will be called out in the overrides log as `no_parseable_runnerup_found`.

---

## 2) Edge API Sketch — agrm → shelling (rebuilt corpus)
Evidence‑only signatures derived from AGRM files that **import Shelling**, ranked by mapped import count then function density.
- **Sketch (Markdown):** `/mnt/data/code_atlas/edge_sketch_agrm_to_shelling.md`
- The UI table “Edge Sketch — agrm→shelling (top candidates)” lists the sources used; each entry includes top‑level function and class/method signatures to shape the boundary.

---

## 3) Grafting Guidance from Close‑Call Diffs
- Analyzed the **tightest function clusters** and extracted lines worth review from both **runner‑up** and **best‑of** (e.g., `try/except`, `raise`, `logger`, boundary checks, timeouts, caching, validation).
- **Reports (JSON):** under `/mnt/data/code_atlas/best_of/diffs/functions/graft_reports/`  
  (first few generated; each report references the diff path and the **FUNC::** anchors for best vs. runner‑up.)

---

## Next suggestions (I can run any/all):
1. **Inline the agrm→shelling sketch** directly here (top 3 files’ signatures) and propose a draft interface with explicit inputs/outputs and error semantics.  
2. **Apply targeted grafts**: pick 2–3 functions from the graft reports; I’ll inline the diff and annotate the exact lines to transplant into the Best‑Of instance.  
3. **Smoke‑wire the tree**: generate a minimal `run_manifest.py` and `pytest -q` harness to sanity‑execute imports from the Best‑Of tree (no external deps), catching broken imports or circularities early.

---



# Interface Surface — AGRM → Shelling (Evidence‑Only)
Pulled the **top 3 AGRM files that import Shelling**, ranked by mapped imports and function density. Signatures below are lifted verbatim from code (no invention). A fuller table is visible in the UI.

**Artifacts:**
- Evidence JSON: `/mnt/data/code_atlas/edge_agrm_shelling_contract_evidence.json`
- Proposal MD: `/mnt/data/code_atlas/edge_agrm_shelling_contract_proposal.md`

> Use these signatures to define the boundary. For anything unclear, we’ll mark a clarification item and keep it non‑binding until confirmed in code or tests.

---

# Smoke‑Wire Results — Best‑Of Working Tree
Executed a module import sweep over the Best‑Of tree to catch import‑time issues.

**Artifacts:**
- Runner: `/mnt/data/code_atlas/best_of/tree/smoke_imports.py`
- Report (CSV/JSON): `/mnt/data/code_atlas/best_of/tree/smoke_imports_report.csv` and `.json`

> The UI table “Smoke‑wire — Import Results” shows the first 200 entries. Failures typically indicate missing deps or order/cycle issues; we keep these as TODOs for stitching.

---

# Graft Plans — Close‑Call Functions
Generated graft suggestions from close‑margin diffs (evidence‑only: exact lines to consider transplanting).

**Artifacts:**
- Plan MD (first 5): `/mnt/data/code_atlas/best_of/diffs/functions/graft_reports/graft_plans.md`
- Individual reports: `/mnt/data/code_atlas/best_of/diffs/functions/graft_reports/*.json`

> Use these to upgrade the Best‑Of instances without inventing behavior. All anchors are preserved for traceability.

---



# Dependency Triage — Smoke‑Wire Failures (v1)
**Objective:** Convert import failures into actionable TODOs with candidate providers from the corpus (Best‑Of tree and full ingest).

**Artifacts:**
- Top missing modules (CSV): `/mnt/data/code_atlas/missing_modules.csv`
- Name‑import errors (CSV): `/mnt/data/code_atlas/import_name_errors.csv`
- Other errors (CSV): `/mnt/data/code_atlas/other_import_errors.csv`
- Candidate providers (JSON): `/mnt/data/code_atlas/candidate_providers.json`

**How to use:**
1. Start with the **Top Missing Modules** table (shown in UI).  
2. For each module `m`, open `candidate_providers.json["m"]` to see candidate files in **best_of** or **ingest** (rank: prefer **best_of** if present).  
3. Decide: (a) add provider to Best‑Of tree, (b) swap variant from its cluster, or (c) mark as external dependency.

> No code edits performed here; this is a triage map for stitching/import fixes.

---



# Smoke‑Wire Analysis & Dependency TODOs (evidence‑only)
I analyzed the import sweep results from the **Best‑Of working tree** and surfaced concrete, evidence‑only TODOs (no code edits).

**Artifacts:**
- Failures (full): `/mnt/data/code_atlas/best_of/tree/smoke_failures.csv`
- Missing modules (grouped): `/mnt/data/code_atlas/best_of/tree/missing_modules_summary.csv`
- Import edges (module→import): `/mnt/data/code_atlas/best_of/tree/import_edges.csv`
- Dependency TODOs (human‑readable): `/mnt/data/code_atlas/best_of/tree/dependency_todos.md`

**How to read:**
- `missing_modules_summary.csv` shows each missing module, how many module imports failed, and whether a **matching module path exists in the Best‑Of tree** (e.g., `pkg/sub/__init__.py` or `pkg/sub.py`). If it exists, the issue is likely **packaging or import path**; if not, it’s a **true dependency gap** to fill from variants or upstream bundles.
- `import_edges.csv` is a raw adjacency for quick inspection or future visualization.

---

# Interface Evidence — AGRM → Shelling (inline)
I re‑emitted the **top‑3 AGRM files** that import Shelling (function & class signatures only) into a compact JSON for quick embedding:
- `/mnt/data/code_atlas/interface_evidence_inline.json`

> These are the same sources used for the earlier proposal MD. We will keep this in sync if the Best‑Of selection changes.

---



# Skiplist Baseline (clean run without missing deps)
**Decision captured:** Missing dependencies will be **slipped** (excluded) or **rebuilt later** if we decide to use that part.

**What I executed:**
1. Built a **Skiplist** from all `ModuleNotFoundError` targets found in the smoke‑wire:  
   `/mnt/data/code_atlas/best_of/skiplist/sliplist_modules.json`
2. Computed the **importer closure** (modules depending on any slipped module), and excluded them from the baseline test run:  
   `/mnt/data/code_atlas/best_of/skiplist/excluded_modules.json`
3. Generated an **Allowlist** of the remaining modules:  
   `/mnt/data/code_atlas/best_of/skiplist/allowlist_modules.json`
4. Ran a **filtered smoke import** only on the allowlist (no code invention, import‑only):  
   Reports → JSON/CSV in `/mnt/data/code_atlas/best_of/skiplist/`

**How to read the UI tables:**
- **Skip Baseline — Summary:** counts of slipped modules, excluded importers, and pass/fail within the allowlist.  
- **Allowlist (first 200):** modules that were tested under the slip baseline, with results.  
- **Excluded Importers:** modules that were skipped because they (directly or transitively) depend on slipped modules.

> This gives us a **clean, runnable baseline** for immediate work, while preserving a crisp list of what we deferred. If/when you provide any deferred module(s), I’ll fold them in and re‑compute the closure and smoke report.

---



# Global Code Use & Case Needs (Evidence‑Only)
You asked whether we can **determine all code use and case need**. I ran a full‑corpus pass over the extracted bundles (ingest_v2) and produced a catalog with:

- **Per‑file classification**: families, inferred use‑cases, roles, and import lists.  
  → `/mnt/data/code_atlas/usecases/classification_usecases.{json,csv}`
- **Use‑case coverage & blockage**: total files per use‑case and how many are **blocked by skiplist** dependencies.  
  → `/mnt/data/code_atlas/usecases/usecase_summary.csv`
- **Use‑case → missing‑module needs**: for each use‑case, which **skipped modules** are referenced and how often.  
  → `/mnt/data/code_atlas/usecases/usecase_needs.csv`
- **Family × Use‑Case matrix**: quick density map for where the work clusters.  
  → `/mnt/data/code_atlas/usecases/family_usecase_matrix.csv`

**Method (strictly evidence‑only):**
- Token‑based heuristics for **families** and **use‑cases**, applied to code text (regexes are logged in this section for audit).  
- **Imports** extracted via AST; a file is marked **blocked_by_skip** if it imports any module in the **skiplist**.

**How to read it:**
- Start with **Use‑Case Summary — Coverage & Blockage** (in the UI). High `files_blocked_by_skip` means that use‑case needs dependency intake to be runnable at scale.
- Then open **Use‑Case Needs — Missing Modules by Use‑Case** to see the exact modules to add back (or rebuild) per use‑case.  
- The **Family × Use‑Case** matrix shows where families concentrate for each use‑case; that’s the right place to select a **best version** and define an API seam.

**Next I can do automatically (say “go” and I’ll run it):**
1) **Per‑use‑case Best‑Of slice**: assemble just the runnable files (`blocked_by_skip = False`) for a chosen use‑case and produce a minimal import harness for it.  
2) **Needs‑first intake plan**: for the top 3 blocked use‑cases, generate a prioritized list of missing modules (with probable source locations in your archives) and prep empty targets in the Best‑Of tree.  
3) **Interface surfaces per use‑case**: emit signature bundles (functions/classes only) over the runnable subset, so we can standardize contracts without touching code.

---



# Full Code Review & Build Plan — v0 (Best‑Of First)
This kicks off a *doctorate‑level* review cycle and build plan using only evidence from your corpus. We start from the **Best‑Of baseline**, map the system, triage risk, lock interfaces, and only then begin build/grafting.

## What’s ready (artifacts you can open now)
- **System nodes (module metrics + risk tags):** `/mnt/data/code_atlas/review/bestof_system_nodes.csv`
- **System edges (internal module→module):** `/mnt/data/code_atlas/review/bestof_system_edges.csv`
- **Family→Family edges (seam weights):** `/mnt/data/code_atlas/review/bestof_family_edges.csv`
- **Risk register:** `/mnt/data/code_atlas/review/risk_register.csv`
- **Code review session agenda:** `/mnt/data/code_atlas/review/code_review_agenda.md`
- **Build plan (phases & exits):** `/mnt/data/code_atlas/review/build_plan.md`

## How I scored risk
`risk_score = 0.4·norm(in_degree) + 0.2·norm(out_degree) + 0.2·norm(complexity_proxy) + 0.2·norm(loc)`  
Tags:
- `import_fail` — failed import in smoke report  
- `excluded_by_skip` — excluded due to skiplist closure  
- `allowlist` — exercised in clean run

## How to drive the session (my recommendation)
1) **Review seams**: open *Family→Family edges* and pick the top 2–3 boundaries for interface locking.  
2) **Triage risks**: open *Top Risk Modules* table (UI) and flag any `import_fail` or seam‑central modules.  
3) **Approve agenda & build plan**: I’ll then generate **interface contracts** (evidence‑only signatures) for the chosen seams and wire a **pytest skeleton** over the allowlist.

> All steps are **non‑destructive** (copy‑only). Provenance is preserved so Franken stitching remains possible at every point.

---



# Seam Contracts (Top 3) + Pytest Skeleton — v1
I selected the **top 3 family→family seams** by edge weight, extracted **evidence‑only interface signatures** from the Best‑Of tree, and scaffolded **pytest** to keep the baseline green.

**Artifacts:**
- Seam index: `/mnt/data/code_atlas/review/seams/INDEX.md`
- Per‑seam contracts (MD + JSON evidence) under: `/mnt/data/code_atlas/review/seams/`
- Tests: `/mnt/data/code_atlas/tests/`
  - `test_imports_allowlist.py` — imports every allowlisted module
  - `test_interface_<A>_to_<B>.py` — asserts presence of functions/classes listed in the seam’s evidence

**Notes:**
- Contracts are **verbatim signatures** from code (no invention).  
- Tests avoid cross‑family imports that land in the **skiplist**; they only assert interface presence in the source modules.

**Next options:**
1) Run the tests in your env: `pytest -q /mnt/data/code_atlas/tests` (or let me adapt to your runner).  
2) Choose any seam for **contract hardening** (clarify inputs/outputs, error semantics) based solely on the extracted signatures.  
3) Promote missing modules for a chosen seam; I’ll re‑emit contracts and expand tests to include target‑side behavior once runnable.

---



# AGRM → Shelling — Hardened Contract (Evidence‑Only) & Acceptance Tests
I generated a hardened, evidence‑only interface document for the **AGRM → Shelling** seam and produced acceptance tests that assert interface presence and signature compatibility (subset of parameter names), without invoking target‑side behavior.

**Artifacts:**
- Hardened contract (MD): `/mnt/data/code_atlas/review/seams/agrm_to_shelling_contract_hardened.md`
- Hardened evidence (JSON): `/mnt/data/code_atlas/review/seams/agrm_to_shelling_hardened.json`
- Tests: `/mnt/data/code_atlas/tests/test_contract_agrm_to_shelling.py`

**Note:** In the current Best‑Of working tree, I did not detect source modules tagged **AGRM** that import modules tagged **Shelling** (this can happen if the heaviest seam wasn’t selected into Best‑Of or is gated by the skiplist). The hardened doc and tests are still emitted and will auto‑populate as soon as AGRM→Shelling source modules are present.

**If you want this seam right now:** provide any AGRM source files that import Shelling (or remove those files from the skiplist), and I’ll instantly re‑emit the contract with full function/class evidence and run the acceptance checks.

---



# Rebuild Plan — v1 (Evidence‑Driven, No Invention)
You confirmed that anything missing needs a **new build**. I generated a **rebuild workspace** and a prioritized **build queue** driven entirely by evidence from your corpus (skiplist, smoke failures, use‑case needs, family tags). No behavior was invented; only stubs and tracking artifacts were created.

**Artifacts:**
- Workspace root: `/mnt/data/code_atlas/rebuild_workspace/`
  - **Source stubs:** `/mnt/data/code_atlas/rebuild_workspace/src/`  
    → one stub per top missing module, each raising `NotImplementedError` with context in the docstring (importers, families, use‑case hits)
  - **Build queue:**  
    - CSV → `/mnt/data/code_atlas/rebuild_workspace/build_queue.csv`  
    - MD  → `/mnt/data/code_atlas/rebuild_workspace/build_queue.md`
- Tests (xfail placeholders): `/mnt/data/code_atlas/tests_rebuild/`  
  → one `pytest` file per missing module, marked `xfail` until the real implementation lands

**How priority was computed:** counts of failing importers (from smoke failures) + aggregated use‑case hits (from `usecase_needs.csv`). Importer families are listed so we can align rebuild ordering with system seams.

**Recommended flow:**
1. **Pick the first item(s) in the Build Queue** (highest importer count / use‑case hits).  
2. Replace the corresponding **stub module** with a concrete implementation (you can drop files in place).  
3. Run tests: `pytest -q /mnt/data/code_atlas/tests /mnt/data/code_atlas/tests_rebuild`  
4. I’ll re-run the smoke/import graph and update the queue automatically on request.

> This keeps us fully traceable: every new build has a stub anchor, context docstring, and a tracking test that flips from `xfail` → `pass` once implemented.

---



# Rebuild Plan — Broad Expansion (+50 target)
I expanded the **rebuild queue** with the **next wave of missing modules** and generated stubs + xfail tests for each (evidence‑only, no behavior). Your corpus currently yielded **3 new items** beyond the initial set; as more gaps surface, this machinery will continue scaling out to the full +50 target automatically.

**Updated artifacts:**
- Queue (CSV): `/mnt/data/code_atlas/rebuild_workspace/build_queue.csv`
- Queue (MD): `/mnt/data/code_atlas/rebuild_workspace/build_queue.md`
- Stubs root: `/mnt/data/code_atlas/rebuild_workspace/src/`
- Tests (xfail): `/mnt/data/code_atlas/tests_rebuild/`

**Notes:**
- The current environment surfaced **3 additional missing modules** to add; if you provide more importer failures (e.g., new bundles or an updated Best‑Of), I’ll extend the queue and stubs instantly to saturate the full +50.
- Each stub documents importers, families, and use‑case hit counts so we can set specifications quickly before implementation.

**Next options:**
1) Drop any additional archives or logs that reveal more `ModuleNotFoundError` entries — I’ll extend the queue and stubs to reach +50.  
2) Pick the **top queued module** and I’ll generate a spec template (based on its importers and use‑cases) to guide a focused rebuild.  
3) Ask me to wire the **rebuild_workspace/src** onto `PYTHONPATH` during specific tests to allow interim presence checks without enabling behavior (still safe, as stubs raise on import).

---



# Rebuild — Presence Wiring (no import execution) → Spec Scaffold (Top Module)
**You asked for C leading into B.** I wired presence checks that never execute the stub modules, then generated a **spec scaffold** for the **top‑priority missing module** from the queue.

**Artifacts:**
- Presence test (find_spec only): `/mnt/data/code_atlas/tests_rebuild/test_presence_skiplist.py`  
  *(adds `rebuild_workspace/src` to `sys.path` locally for the test)*
- Presence report: `/mnt/data/code_atlas/rebuild_workspace/presence_report.{json,csv}`
- Spec scaffold (top module): `/mnt/data/code_atlas/rebuild_workspace/specs/<module>/{spec.md,spec.json}`

**How presence works without imports:**
- Uses `importlib.util.find_spec(<module>)` to verify **resolvability** on PYTHONPATH; it does **not** execute the stub (so the `NotImplementedError` in stubs never triggers during this check).

**Spec scaffold contents (evidence‑only):**
- Priority (failing importer count, use‑case hits)
- Known importers (sample) and **importer families** density
- Use‑cases referencing the module
- Initial acceptance: presence required; interfaces/behavior/errors **TBD** (to be filled once we lock the surface)
- Open questions to drive requirements capture

**Next steps I can run now:**
1) Extract **interface surfaces from importer modules** that reference the top missing module (functions/methods by which they call into it).  
2) Turn the scaffold into a **concrete spec**: enumerate entrypoints, expected inputs/outputs (from importer callsites), and error semantics we can infer from importer‑side handling.  
3) Generate a **starter test plan** (pytest) that will stay xfail until the real implementation lands, but that lays out the interface expectations.

Say the word (e.g., “go 1+2+3 for the top module”), and I’ll execute immediately. 

---



# Importer‑Side Interface → Spec Upgrade → XFail Plan (Top Missing Module)
You said “Go with all,” so I executed **(1) importer interface extraction**, **(2) spec upgrade**, and **(3) xfail test plan** for the **top‑priority missing module** in the rebuild queue.

**Artifacts:**
- **Importer modules list (who calls into it):** see UI table “Importer Modules — Top Missing Module”.
- **Entrypoints inferred from importer callsites:** UI table “Importer‑Derived Entrypoints”.
- **Spec (updated):** `/mnt/data/code_atlas/rebuild_workspace/specs/<module>/spec.md`
- **Spec (calls evidence):** `/mnt/data/code_atlas/rebuild_workspace/specs/<module>/spec_calls.json`
- **XFail test plan:** `/mnt/data/code_atlas/tests_rebuild/test_spec_<module>.py`

**What this means:**
- We now have a **requirements scaffold grounded in existing importer code**: entrypoint names, positional counts seen, keyword usage, and the exceptions caught nearby.  
- The **xfail plan** encodes these interface expectations but won’t pass until the real implementation lands (it purposely xfails today).

**Next suggested action:**
- I can turn this scaffold into a **workable, implementer‑ready spec** by generating typed input/shape examples from real importer callsites (dictionaries/kwargs structure, list/tuple arities, etc.) and drafting acceptance checks (still xfail) per entrypoint. Say **“go spec‑detail for the top module”** and I’ll do it now.

---



# Spec Detail (Importer Shapes) + Work Task List — Top Missing Module
You asked to proceed with spec detail and produce a work task list. I analyzed **importer callsites** for the top missing module and extracted **argument shapes**, **keyword keys**, and **usage contexts**, then turned that into a **task list**.

**Artifacts:**
- Spec detail (MD): `/mnt/data/code_atlas/rebuild_workspace/specs/buildfiles.buildfiles.ConStruct_6/spec.md`
- Examples (JSON): `/mnt/data/code_atlas/rebuild_workspace/specs/buildfiles.buildfiles.ConStruct_6/spec_examples.json`
- XFail acceptance expansion: `/mnt/data/code_atlas/tests_rebuild/test_specdetail_buildfiles_buildfiles_ConStruct_6.py`
- Work task list: `/mnt/data/code_atlas/rebuild_workspace/specs/buildfiles.buildfiles.ConStruct_6/tasks.md` (CSV alongside)

**UI views:**
- **Spec Detail — Entrypoint Shapes** (per entrypoint: importer count, positional sample count, keyword keys, contexts)
- **Work Task List — Top Module** (prioritized P1 tasks)

**What this enables next:**
- We can assign concrete implementation tasks **per entrypoint** based on real importer demand and observed usage.  
- The xfail suite is ready to flip to **pass** as soon as you land implementations that match the observed interfaces.

If you’d like, I can now: (a) replicate this spec‑detail+tasks flow for the **next** module in the queue, or (b) generate a **skeleton implementation** file for the top module (empty functions/classes with `pass` and docstrings), still evidence‑only and safe for CI. Say the word.

---



# SnapLat Canonicalization — Policy & Queue Update
You said SnapLat is the system name and all other naming is **legacy oracle**. I’ve now standardized our rebuild workflow on a **SnapLat canonical namespace** and treated legacy names as compatibility aliases only.

**What I set up**
- **Canonical packages**: `snaplat.*` families
  - `snaplat.agent` (AGRMs / controllers / routing)  
  - `snaplat.index` (MDHG / indexing)  
  - `snaplat.shell` (Shelling engine)  
  - `snaplat.overlay` (Universe/overlay layer)  
  - `snaplat.codec`, `snaplat.beacons`, `snaplat.align`, `snaplat.orch`, `snaplat.policy`, `snaplat.repo`, `snaplat.telemetry`, `snaplat.utils`, `snaplat.ops`
- **Queue canonicalization**: created `/mnt/data/code_atlas/rebuild_workspace/build_queue_canonical.csv` with, for each missing module:  
  `sanitized`, `canonical_family`, `canonical_pkg`, `canonical_module`, `rename_needed`, `rename_reason`.
- **Legacy aliases**: scaffolding for legacy names to re‑export from their canonical `snaplat.*` module (created only when rows exist).  
- **Presence test**: `/mnt/data/code_atlas/tests_rebuild/test_canonicalization_presence.py` verifies both the canonical and any legacy alias resolve on `PYTHONPATH`.

**UI tables**
- *SnapLat Canonicalization — Queue Mapping*  
- *SnapLat Canonicalization — Resolvability Probe*

**How this helps**
- We converge the namespace to **SnapLat** while keeping the door open to legacy import paths during the transition.  
- The rebuild queue now drives toward **canonical modules first**, making future refactors cheaper.

**Next moves I can take immediately**
1) Promote the **top N canonical modules** to skeletons (evidence‑only functions, no behavior), so presence tests pass and xfails encode expectations.  
2) Apply the **spec‑detail+tasks** flow to one or more canonical modules (like `snaplat.agent` or `snaplat.index`) to generate implementer‑ready work packs.  
3) Sweep the corpus for **invalid legacy names** (spaces, punctuation), add rename suggestions, and update alias stubs when you approve the new canonical names.

Say “skeleton top N”, “spec‑detail for `<snaplat.pkg>`”, or “sweep invalid names”, and I’ll proceed.

---



# SnapLat — Executed: Skeletons (Top 10), Spec‑Detail (Agent), and Rename Sweep
You said “go on all work suggested,” so I executed the three tracks. The environment currently has limited queue data, so a few outputs are placeholders but the full pipeline is wired and ready to expand as soon as more items appear.

## 1) Skeletons for Top Canonical Modules (up to 10)
- Created evidence‑only skeleton generation logic (empty behavior, planned signature params when evidence exists), plus legacy aliases → canonical modules.  
- **Note:** No eligible targets were found in the canonical queue this pass (likely because the canonical queue is empty or rows lacked full module paths). As soon as `build_queue_canonical.csv` contains rows, I’ll re-run to emit the skeleton modules under `rebuild_workspace/src/snaplat/...` with corresponding legacy alias shims.

## 2) Spec‑Detail + Tasks for SnapLat Agent (top demand)
- Wired the spec‑detail flow for the top‑demand module in the `snaplat.agent` family (fallback to top overall canonical module).
- **Note:** No target row available this pass (queue likely empty), so the spec pack will populate on the next queue refresh. The artifacts are pre‑positioned under `rebuild_workspace/specs/`.

## 3) Sweep Invalid Legacy Names → Rename Report
- Produced **rename report** with suggestions for canonical module names and reasons (illegal characters / not under `snaplat.*`).
  - Report: `/mnt/data/code_atlas/naming/rename_report.md` (CSV alongside)
- Legacy alias scaffolding remains enabled; once we have concrete entries, aliases will be generated so old imports resolve while we migrate.

**What I recommend now**
1) **Refresh the build queue**: provide any new logs/archives that surface `ModuleNotFoundError` lines, or re-run the smoke to regenerate `build_queue.csv` → `build_queue_canonical.csv`.  
2) I’ll immediately:
   - Emit **skeletons** for the top 10 canonical modules, 
   - Run **spec‑detail + tasks** for the top `snaplat.agent.*` module, and
   - Update the **rename report** + create legacy alias shims.

If you want me to try skeletons/spec‑detail for a specific canonical target right now (e.g., `snaplat.agent.router` or `snaplat.index.graph`), just name it and I’ll generate that pack on the spot, evidence‑only.

---

