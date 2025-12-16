# ADR-0002: Quantile + EMA Tuning

**Decision:** Tune promote/demote via hit quantiles; smooth via EMA.

**Context:** Avoid oscillation; reflect actual hit distribution under skew.

**Consequences:** Slower but stable convergence; reduced thrash during phase transitions.
