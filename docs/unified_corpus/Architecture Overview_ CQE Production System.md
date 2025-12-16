# Architecture Overview: CQE Production System

## Core Philosophy

The CQE system is built on the principle that computation is fundamentally geometric. Its architecture is designed to exploit the inherent geometric symmetries of data and problems to achieve optimal solutions.

## High-Level Components

```
+---------------------------------------+
|           main.py (Entry Point)       |
+---------------------------------------+
|    Aletheia AI (Local Reasoning)      |
+---------------------------------------+
|      Tools (SpeedLight, Governance)   |
+---------------------------------------+
|      Core (Lattices, Conservation)    |
+---------------------------------------+
| Data (MonsterMoonshineDB, RAG Cards)  |
+---------------------------------------+
```

### 1. Core Engine (`/core`)

-   **Function**: Provides the fundamental mathematical and geometric primitives.
-   **Key Modules**:
    -   `e8_lattice.py`: Operations on the E8 lattice.
    -   `niemeier_lattices.py`: Definitions and properties of all 24 Niemeier lattices.
    -   `conservation.py`: Enforcement of the conservation law (ΔΦ ≤ 0).
    -   `weyl_cartan.py`: Weyl group reflections and Cartan matrix operations.
    -   `monster_voa.py`: Mappings to the Monster Group and Moonshine VOA.

### 2. Tool Suite (`/tools`)

-   **Function**: High-level tools that use the core engine to perform complex analysis.
-   **Key Modules**:
    -   `speedlight.py`: The equivalence class detection and caching engine. The heart of the system's efficiency.
    -   `ca_tiles.py`: Cellular automata for pattern and density analysis.
    -   `lattice_viewer.py`: Simulates problem perspectives from different lattice viewpoints.
    -   `governance.py`: The meta-governance engine for responsible AI operation.

### 3. Aletheia AI (`/aletheia`)

-   **Function**: A local, stdlib-only AI reasoning engine.
-   **Key Modules**:
    -   `reasoning_engine.py`: Orchestrates analysis and synthesis.
    -   `rag.py`: Retrieval-Augmented Generation using the RAG cards.
    -   `embeddings.py`: Interacts with MonsterMoonshineDB for geometric lookups.

### 4. Data Layer (`/data`)

-   **Function**: The knowledge base of the system.
-   **Key Components**:
    -   `monster_moonshine_db/`: The 24D embedding database. Contains embeddings for all session work and concepts.
    -   `rag_cards/`: 40 individual knowledge cards in JSON format.
    -   `relationships.csv`: A graph of 312 connections between key concepts.

## Data Flow: The 8-Fold Path

A typical request follows the 8-fold path of maximal organization:

1.  **State Definition**: Prune solution space.
2.  **Lattice Simulation**: Simulate 24 perspectives.
3.  **CA Analysis**: Analyze data density.
4.  **SpeedLight**: Find equivalence classes.
5.  **Embedding Retrieval**: Look up similar problems in MonsterMoonshineDB.
6.  **Governance**: Evaluate ethics and responsibility.
7.  **Synthesis**: Aletheia AI synthesizes a unified solution.
8.  **Meta-Observer**: The process itself is recognized as the solution (toroidal wrap).

This architecture ensures that every solution is geometrically optimal, computationally efficient, and ethically sound.
