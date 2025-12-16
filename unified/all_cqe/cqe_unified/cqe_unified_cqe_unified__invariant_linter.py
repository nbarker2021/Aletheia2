
from typing import Dict, Any, List

class GeometryFirstLinter:
    """Static linter for geometry-first compliance in receipts."""
    required_fields = ["stage_order","e8_embedding","lattice_ops","semantics","parity_channels","constraints"]
    required_order_prefix = ["geometric_encoding","e8_embedding","lattice_ops"]

    def check_receipt(self, receipt: Dict[str,Any]) -> List[str]:
        errs = []
        for f in self.required_fields:
            if f not in receipt:
                errs.append(f"missing field: {f}")
        # order check
        order = receipt.get("stage_order", [])
        if order[:3] != self.required_order_prefix:
            errs.append("stage order must start with geometry-first triad")
        # parity shape
        pc = receipt.get("parity_channels", [])
        if not isinstance(pc, list) or len(pc) != 8:
            errs.append("parity_channels must be length-8 list")
        # constraints presence
        if "weyl" not in str(receipt.get("constraints", "")).lower():
            errs.append("constraints must mention Weyl/chamber info")
        return errs
