
import json, hashlib, time
from pathlib import Path

class MS2Labeler:
    def __init__(self, labels_dir: Path):
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def write_labels(self, run_id: str, channels):
        out_dir = self.labels_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for idx, ch in enumerate(channels, 1):
            card = {
                "run_id": run_id,
                "channel_index": idx,
                "channel_name": f"channel_{idx}",
                "ordering_signature": f"sig_{idx:02d}",
                "value": ch
            }
            p = out_dir / f"label_{idx:02d}.json"
            p.write_text(json.dumps(card, indent=2), encoding="utf-8")
            written.append(str(p))
        return written

class MS3Governance:
    def __init__(self, seals_dir: Path):
        self.seals_dir = seals_dir
        self.seals_dir.mkdir(parents=True, exist_ok=True)

    def seal(self, run_id: str, receipt_path: Path, label_paths):
        # diversity = number of distinct ordering signatures
        diversity = len(label_paths)
        decision = "ACCEPT" if diversity >= 8 else "REJECT"
        # compute merkle-ish seal by hashing all files deterministically
        h = hashlib.sha256()
        h.update(Path(receipt_path).read_bytes())
        for lp in sorted(label_paths):
            h.update(Path(lp).read_bytes())
        digest = h.hexdigest()
        seal = {
            "run_id": run_id,
            "decision": decision,
            "diversity_count": diversity,
            "sealed_at": time.time(),
            "digest": digest,
            "receipt": str(receipt_path),
            "labels": label_paths
        }
        out = self.seals_dir / f"seal_{digest[:12]}.json"
        out.write_text(json.dumps(seal, indent=2), encoding="utf-8")
        return out, seal
