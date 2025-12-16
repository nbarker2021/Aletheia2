# Security Policy

## Threat Model
- Identify assets: code, data, APIs
- Enumerate threats: data exfiltration, dependency vulnerabilities, unauthorized API access

## Data Handling
- Encryption at rest (AES-256)
- TLS 1.2+ for data in transit
- Access controls: RBAC, least privilege

## Dependency Audits
- Automated `pip-audit` and `safety` scans on each CI run
- SBOM generation and review

## Incident Response
- SLA: 24h to acknowledge, 72h to mitigate
- Security contact: security@example.com

# Versioning Policy

## Semantic Versioning
- Format: MAJOR.MINOR.PATCH
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

## Deprecation
- Deprecation announced one minor version before removal
- Maintain backward compatibility during deprecation period

## LTS and Backports
- LTS branches every MAJOR release
- Backport critical bug and security fixes to LTS

# Release Management

## Pre-Release Gates
- Static analysis: `flake8`, `mypy`
- Tests: unit coverage ≥90%, integration ≥80%
- Dependency audit: no high-severity findings
- SBOM scan: license compliance

## Release Steps
1. Tag annotated release: `vMAJOR.MINOR.PATCH`
2. Update changelog with highlights
3. Sign source and binaries (GPG)
4. Verify rollback plan documented
5. Publish artifacts to package registry

# API Stability

## Versioned Endpoints
- Prefix: `/v1/`, `/v2/`, etc.
- Deprecated endpoints marked in docs and removed after deprecation period

## Contract Tests
- Use consumer-driven contract tests (e.g., Pact) to validate backwards compatibility

## Feature Flags
- Toggle experimental endpoints via config
- Remove flags after stabilization

