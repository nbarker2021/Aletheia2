# P2.5: Public Dataset Publication & Replication Platform Package

This package publishes all datasets and provides guidelines for a replication platform.

## Artifacts

1. **Zenodo Metadata JSON**: `zenodo_deposition.json`
2. **Replication Dockerfile**: `Dockerfile`
3. **Replication Pipeline Config**: `.github/workflows/replicate.yml`
4. **Publication Report PDF**: `dataset_publication_report.pdf`

## zenodo_deposition.json
```json
{
  "title": "CQE Phase 0-4 Datasets",
  "upload_type": "dataset",
  "description": "Includes boundary coordinates, DQPT series, modular coefficients, ablation results, and integration artifacts.",
  "creators": [{"name": "Research Team", "affiliation": "Perplexity AI"}],
  "keywords": ["CQE", "Niemeier", "Dynamical Quantum Phase Transitions", "Modular Forms"],
  "access_right": "open",
  "license": "CC-BY-4.0"
}
```

## Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install numpy scipy matplotlib mpmath sympy requests lxml h5py sklearn
CMD ["bash","-c","python phase2_artifacts.md && python phase4_stats.py"]
```

## .github/workflows/replicate.yml
```yaml
name: Replication CI
on: [push]
jobs:
  replicate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: docker build -t cqe-replica .
      - run: docker run --rm cqe-replica
```