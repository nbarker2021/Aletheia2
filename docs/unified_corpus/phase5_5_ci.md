# Phase 5.5: CI/CD Pipeline Config

This YAML defines automated builds, tests, and deployments for all CQE components.

```yaml
name: CQE CI/CD Pipeline
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: 3.10
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Lint and type check
        run: |
          flake8 lib api src
          mypy lib api src
      - name: Run tests
        run: |
          pytest --maxfail=1 --disable-warnings -q
      - name: Build Docker images
        run: |
          docker build -t cqe-cli .
          docker build -t cqe-api ./api

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    steps:
      - uses: actions/checkout@v2
      - name: Publish API image
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USER }} --password-stdin
          docker push cqe-api:latest
      - name: Publish CLI image
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USER }} --password-stdin
          docker push cqe-cli:latest
```