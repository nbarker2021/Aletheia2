# Extracted from: CQE_CORE_MONOLITH.py
# Class: SliceSensors
# Lines: 110

class SliceSensors:
    def __init__(self, W: int = 80, topk: int = 16):
        self.W = W
        self.topk = topk
        self.theta = [2.0 * math.pi * m / W for m in range(W)]

    # --- projections & helpers ---
    @staticmethod
    def _project_stream(vals: Sequence[int], base: int, theta: float) -> List[float]:
        # Treat each sample as a point on its base-gon; project onto direction 
        out: List[float] = []
        for v in vals:
            ang = 2.0 * math.pi * (v % base) / base
            out.append(math.cos(ang - theta))
        return out

    @staticmethod
    def _argmax_idx(arr: Sequence[float]) -> int:
        best = -1e9; idx = 0
        for i, x in enumerate(arr):
            if x > best:
                best = x; idx = i
        return idx

    @staticmethod
    def _quadrant_bins(vals: Sequence[int], base: int, theta: float) -> Tuple[int,int,int,int]:
        # Bin positions after rotation; 4 equal arcs on the circle
        bins = [0,0,0,0]
        for v in vals:
            ang = (2.0 * math.pi * (v % base) / base - theta) % (2.0 * math.pi)
            q = int((ang / (2.0 * math.pi)) * 4.0) % 4
            bins[q] += 1
        return (bins[0], bins[1], bins[2], bins[3])

    @staticmethod
    def _chord_hist(vals: Sequence[int], base: int) -> Dict[int,int]:
        c: Dict[int,int] = {}
        for a, b in zip(vals, vals[1:]):
            step = (b - a) % base
            c[step] = c.get(step, 0) + 1
        return c

    @staticmethod
    def _perm_by_projection(vals: Sequence[int], base: int, theta: float, topk: int) -> List[int]:
        proj = SliceSensors._project_stream(vals, base, theta)
        order = sorted(range(len(vals)), key=lambda i: proj[i], reverse=True)
        return order[:min(topk, len(order))]

    @staticmethod
    def _adjacent_transpositions(prev: List[int], curr: List[int]) -> int:
        # Count inversions between adjacent elements moving from prev to curr (small topk, O(n^2) ok)
        pos_curr = {v: i for i, v in enumerate(curr)}
        common = [v for v in prev if v in pos_curr]
        mapped = [pos_curr[v] for v in common]
        inv = 0
        for i in range(len(mapped)):
            for j in range(i+1, len(mapped)):
                if mapped[i] > mapped[j]:
                    inv += 1
        return inv

    def compute(self, face: Face) -> SliceObservables:
        W, base, vals = self.W, face.base, face.values
        theta = self.theta
        extreme_idx: List[int] = []
        quadrant_bins: List[Tuple[int,int,int,int]] = []
        chord_hist: List[Dict[int,int]] = []
        perm: List[List[int]] = []
        braid_current: List[int] = []

        prev_order: Optional[List[int]] = None
        for th in theta:
            proj = self._project_stream(vals, base, th)
            extreme_idx.append(self._argmax_idx(proj))
            quadrant_bins.append(self._quadrant_bins(vals, base, th))
            chord_hist.append(self._chord_hist(vals, base))  # independent of  in this simple model
            order = self._perm_by_projection(vals, base, th, self.topk)
            perm.append(order)
            if prev_order is None:
                braid_current.append(0)
            else:
                braid_current.append(self._adjacent_transpositions(prev_order, order))
            prev_order = order

        # Energies (Dirichlet) on discrete circle
        def dirichlet_energy_int(seq: Sequence[int]) -> float:
            n = len(seq); acc = 0.0
            for i in range(n):
                a = seq[(i+1) % n]; b = seq[i]; c = seq[(i-1) % n]
                acc += float((a - 2*b + c)**2)
            return acc / float(max(1, n))

        def q_imbalance_energy(qbins: Sequence[Tuple[int,int,int,int]]) -> float:
            e = 0.0
            for q in qbins:
                m = sum(q) / 4.0
                e += sum((qi - m)**2 for qi in q)
            return e / float(max(1, len(qbins)))

        energies = {
            "E_extreme": dirichlet_energy_int(extreme_idx),
            "E_quads": q_imbalance_energy(quadrant_bins),
            "Crossings": float(sum(braid_current)),
        }
        return SliceObservables(theta, extreme_idx, quadrant_bins, chord_hist, perm, braid_current, energies)

# -----------------------------------------------------------------------------
# Actuators
# -----------------------------------------------------------------------------
