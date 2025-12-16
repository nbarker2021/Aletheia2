# SNAP — Selective Segment Plan (v1)

Decision categories: **Adopt (segment)**, **Idea-Only Reimplementation**, **Reference Only**.

## 1. SNAPRecord
- Code: `/mnt/data/_bestof_unpack/bestof_agrm/snap/__init__.py`
- Docs:
  - `/mnt/data/_bestof_unpack/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest (1).md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest.md`
  - `/mnt/data/SNAP_Triage_Report_20250811.md`
- Exports: (implicit)
- Reusability: segment-ready • Risk: low • Deps: none • Import: n/a
- **Action:** Adopt (selective segment)
- Why: Small, low-risk segment; fits interfaces
- Snippet: …range(len(items)) ] composite.sort(key=lambda t: t[0], reverse=True) return composite[:k] ``` --- ## bestof_agrm/snap/__init__.py ```python """SNAP public API.""" from .core import SNAP, SNAPRecord, SNAPType, SnapshotID, SNAPError from .types import SNAPToken, SNAPDoc, SNAPKernel, [...]…

## 2. SNAPType
- Code: `/mnt/data/_bestof_unpack/bestof_agrm/snap/__init__.py`
- Docs:
  - `/mnt/data/_bestof_unpack/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest (1).md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest.md`
  - `/mnt/data/SNAP_Triage_Report_20250811.md`
- Exports: (implicit)
- Reusability: segment-ready • Risk: low • Deps: none • Import: n/a
- **Action:** Adopt (selective segment)
- Why: Small, low-risk segment; fits interfaces
- Snippet: …ems)) ] composite.sort(key=lambda t: t[0], reverse=True) return composite[:k] ``` --- ## bestof_agrm/snap/__init__.py ```python """SNAP public API.""" from .core import SNAP, SNAPRecord, SNAPType, SnapshotID, SNAPError from .types import SNAPToken, SNAPDoc, SNAPKernel, register_snap_type, [...]…

## 3. SNAPError
- Code: `/mnt/data/_bestof_unpack/bestof_agrm/snap/__init__.py`
- Docs:
  - `/mnt/data/_bestof_unpack/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/bestof_agrm_source_listing.md`
  - `/mnt/data/SNAP_Triage_Report_20250811.md`
  - `/mnt/data/SNAPDNA_Focused_Search_20250811.json`
  - `/mnt/data/SNAPDNA_Focused_Findings_20250811.md`
- Exports: (implicit)
- Reusability: segment-ready • Risk: low • Deps: none • Import: n/a
- **Action:** Adopt (selective segment)
- Why: Small, low-risk segment; fits interfaces
- Snippet: …sort(key=lambda t: t[0], reverse=True) return composite[:k] ``` --- ## bestof_agrm/snap/__init__.py ```python """SNAP public API.""" from .core import SNAP, SNAPRecord, SNAPType, SnapshotID, SNAPError from .types import SNAPToken, SNAPDoc, SNAPKernel, register_snap_type, get_snap_type from [...]…

## 4. SNAPToken
- Code: `/mnt/data/_bestof_unpack/bestof_agrm/snap/__init__.py`
- Docs:
  - `/mnt/data/_bestof_unpack/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest (1).md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest.md`
  - `/mnt/data/SNAP_Triage_Report_20250811.md`
- Exports: (implicit)
- Reusability: segment-ready • Risk: low • Deps: none • Import: n/a
- **Action:** Adopt (selective segment)
- Why: Small, low-risk segment; fits interfaces
- Snippet: …rse=True) return composite[:k] ``` --- ## bestof_agrm/snap/__init__.py ```python """SNAP public API.""" from .core import SNAP, SNAPRecord, SNAPType, SnapshotID, SNAPError from .types import SNAPToken, SNAPDoc, SNAPKernel, register_snap_type, get_snap_type from .storage import FileStorage, [...]…

## 5. SNAPDoc
- Code: `/mnt/data/_bestof_unpack/bestof_agrm/snap/__init__.py`
- Docs:
  - `/mnt/data/_bestof_unpack/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest (1).md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest.md`
  - `/mnt/data/SNAP_Triage_Report_20250811.md`
- Exports: (implicit)
- Reusability: segment-ready • Risk: low • Deps: none • Import: n/a
- **Action:** Adopt (selective segment)
- Why: Small, low-risk segment; fits interfaces
- Snippet: …return composite[:k] ``` --- ## bestof_agrm/snap/__init__.py ```python """SNAP public API.""" from .core import SNAP, SNAPRecord, SNAPType, SnapshotID, SNAPError from .types import SNAPToken, SNAPDoc, SNAPKernel, register_snap_type, get_snap_type from .storage import FileStorage, MemoryStorage [...]…

## 6. SNAPKernel
- Code: `/mnt/data/_bestof_unpack/bestof_agrm/snap/__init__.py`
- Docs:
  - `/mnt/data/_bestof_unpack/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/bestof_agrm_source_listing.md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest (1).md`
  - `/mnt/data/_older_unpack/older versions and other files/targeted_digest.md`
  - `/mnt/data/SNAP_Triage_Report_20250811.md`
- Exports: (implicit)
- Reusability: segment-ready • Risk: low • Deps: none • Import: n/a
- **Action:** Adopt (selective segment)
- Why: Small, low-risk segment; fits interfaces
- Snippet: …composite[:k] ``` --- ## bestof_agrm/snap/__init__.py ```python """SNAP public API.""" from .core import SNAP, SNAPRecord, SNAPType, SnapshotID, SNAPError from .types import SNAPToken, SNAPDoc, SNAPKernel, register_snap_type, get_snap_type from .storage import FileStorage, MemoryStorage from [...]…
