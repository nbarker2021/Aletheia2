# ADR-0001: Hierarchical Hash vs Flat Hash

**Decision:** Use hierarchical Building→Floor→Room layout with φ-scale growth and promotion tiers.

**Context:** Skewed hot keys and shifting workloads require sustained tail-latency and cheap resizing.

**Consequences:** Slight complexity increase; improved locality and stable load under bursts.
