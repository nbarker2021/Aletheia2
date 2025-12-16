# Data Schemas (JSON) — v0

> Minimal JSON Schema-style outlines for key SNAP types.

## ArXivPaper
```json
{ "type":"object", "required":["kind","id","title","url"],
  "properties": {"kind":{"const":"ArXivPaper"}, "id":{"type":"string"},
    "title":{"type":"string"}, "url":{"type":"string"}, "authors":{"type":"array","items":{"type":"string"}} } }
```

## PaperDigest
```json
{ "type":"object", "required":["kind","paper_id"],
  "properties": {"kind":{"const":"PaperDigest"}, "paper_id":{"type":"string"},
    "problem":{"type":"string"}, "method":{"type":"array","items":{"type":"string"}},
    "key_findings":{"type":"array","items":{"type":"string"}} } }
```

## LLMProfile
```json
{ "type":"object", "required":["kind","id"],
  "properties": {"kind":{"const":"LLMProfile"}, "id":{"type":"string"},
    "context":{"type":"integer"}, "tools":{"type":"array","items":{"type":"string"}},
    "evals":{"type":"object"} } }
```

## AlgorithmCard
```json
{ "type":"object", "required":["kind","name"],
  "properties": {"kind":{"const":"AlgorithmCard"}, "name":{"type":"string"},
    "steps":{"type":"array","items":{"type":"string"}}, "knobs":{"type":"object"} } }
```

## CodeUnit (SNAPBIT)
```json
{ "type":"object", "required":["kind","id","lang","signature","body"],
  "properties": {"kind":{"const":"CodeUnit"}, "id":{"type":"string"},
    "lang":{"type":"string"}, "signature":{"type":"object"},
    "deps":{"type":"array","items":{"type":"string"}}, "tests":{"type":"array","items":{"type":"string"}} } }
```

## ErrorTrace
```json
{ "type":"object", "required":["kind","id","exception"],
  "properties":{"kind":{"const":"ErrorTrace"}, "id":{"type":"string"},
    "exception":{"type":"string"}, "stack":{"type":"string"},
    "tags":{"type":"array","items":{"type":"string"}}, "repro":{"type":"object"} } }
```

## PlaybookStep
```json
{ "type":"object", "required":["kind","id","when","do"],
  "properties":{"kind":{"const":"PlaybookStep"}, "id":{"type":"string"},
    "when":{"type":"array","items":{"type":"string"}},
    "do":{"type":"array","items":{"type":"string"}},
    "verify":{"type":"array","items":{"type":"string"}} } }
```

## SNAPDNA
```json
{ "type":"object", "required":["kind","id","expertise"],
  "properties":{"kind":{"enum":["SNAPDNA","HybridSNAPDNA"]},"id":{"type":"string"},
    "expertise":{"type":"object"}, "routing_tags":{"type":"array","items":{"type":"string"}},
    "tools":{"type":"array","items":{"type":"string"}}, "score":{"type":"number"} } }
```
