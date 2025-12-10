class CQEController:
    def __init__(self, policy: Policy, out_dir: Path):
        self.policy = policy
        self.out = out_dir
        self.writer = ReceiptWriter(out_dir)

    # --- core loop on a single face ---
    def normalize_face(self, face: Face, channel: str, idx_range: Tuple[int,int]=(0,0)) -> Dict[str, Any]:
        pol = self.policy
        best: Optional[Dict[str, Any]] = None
        # Try repair OFF/ON and lattices (80 then 240)
        for repair_flag in (False, True):
            for W in pol.lattice_candidates:
                sens = SliceSensors(W=W)
                vals = list(face.values)
                rep_info: Dict[str, Any] = {"edits": 0}
                if repair_flag:
                    vals, rep_info = Actuators.least_action_repair(vals, face.base)
                obs = sens.compute(Face(vals, face.base, face.label))

                # Equalizer: choose θ index with minimal quadrant variance at that θ
                q_var = []
                for qb in obs.quadrant_bins:
                    m = sum(qb)/4.0
                    q_var.append(sum((x-m)**2 for x in qb))
                theta_star_idx = min(range(W), key=lambda i: q_var[i])
                theta_deg = 360.0 * theta_star_idx / W

                # Keys and objective
                d10_key = Keys.delta_key(Face(vals, 10, "decagon"))
                d8_key  = Keys.delta_key(Face(vals, 8, "octagon"))
                pose_key = Keys.pose_key_W(Face(vals, face.base, face.label), obs)
                J = Objective.J(pol, obs, d10_key, d8_key, rep_info, pose_key)

                candidate = {
                    "theta_deg": theta_deg,
                    "W": W,
                    "repair": repair_flag,
                    "clones_K": Actuators.minK_to_balance(obs.quadrant_bins),
                    "obs": obs,
                    "rep_info": rep_info,
                    "d10_key": d10_key,
                    "d8_key": d8_key,
                    "pose_key": pose_key,
                    "J": J,
                    "vals": vals,
                }
                if (best is None) or (candidate["J"] < best["J"]):
                    best = candidate
        assert best is not None

        # Validators (stubs for now)
        gates = {
            "ΔΦ": True,
            "LATT": Validators.latt_stub(face).ok,
            "CRT": Validators.crt_stub(face).ok,
            "FRAC": Validators.frac_stub(best["obs"]).ok,
            "SACNUM": Validators.sacnum_stub(face).ok,
        }

        # Receipt
        pre = {"J": best["J"], "theta": best["theta_deg"], "W": best["W"], "repair": best["repair"], "K": best["clones_K"]}
        post = dict(pre)  # single step
        energies = best["obs"].energies
        writhe = int(sum(best["obs"].braid_current))
        braid = {"writhe": writhe, "crossings": writhe, "windows": []}
        parity64 = hashlib.sha256((channel + str(idx_range) + str(best["vals"])).encode()).hexdigest()[:16]
        pose_salt = hashlib.md5(best["pose_key"].encode()).hexdigest()[:8]
        merkle = {"path": sha256_hex([pre, post, energies, braid])[:32]}
        rec = Receipt(
            claim="CQE.normalize",
            pre=pre, post=post,
            energies=energies,
            keys={"pose_W80": best["pose_key"], "d10": best["d10_key"], "d8": best["d8_key"], "joint": Keys.joint_key(best["d10_key"], best["d8_key"])},
            braid=braid,
            validators=gates,
            parity64=parity64,
            pose_salt=pose_salt,
            merkle=merkle,
        )
        self.writer.append_ledger(rec)

        # LPC row
        lpc = LPCRow(
            face_id=sha256_hex([channel, idx_range]),
            channel=channel,
            idx_range=idx_range,
            equalizing_angle_deg=best["theta_deg"],
            pose_key_W80=best["pose_key"],
            d10_key=best["d10_key"],
            d8_key=best["d8_key"],
            joint_key=Keys.joint_key(best["d10_key"], best["d8_key"]),
            writhe=writhe,
            crossings=writhe,
            clone_K=best["clones_K"],
            quad_var_at_eq=float(energies.get("E_quads", 0.0)),
            repair_family_id="odd-coprime@base",
            residues_hash=sha256_hex(best["vals"]),
            proof_hash=merkle["path"],
        )
        # Write LPC
        with open(self.writer.lpc_path, "a", encoding="utf-8") as f:
            f.write("|".join([
                lpc.face_id, lpc.channel, str(lpc.idx_range[0]), str(lpc.idx_range[1]), f"{lpc.equalizing_angle_deg:.6f}",
                lpc.pose_key_W80, lpc.d10_key, lpc.d8_key, lpc.joint_key, str(lpc.writhe), str(lpc.crossings),
                str(lpc.clone_K), f"{lpc.quad_var_at_eq:.6f}", lpc.repair_family_id, lpc.residues_hash, lpc.proof_hash
            ]) + "\n")

        return {
            "state": {k: best[k] for k in ("theta_deg","W","repair","clones_K")},
            "energies": energies,
            "keys": rec.keys,
            "validators": gates,
            "receipt_hash": rec.merkle["path"],
        }

    # High-level convenience
    def normalize(self, text: str) -> Dict[str, Any]:
        dec, octv = text_to_faces(text)
        out = {"policy": dc.asdict(self.policy), "faces": {}}
        out["faces"]["decagon"] = self.normalize_face(dec, channel="decagon", idx_range=(0, len(dec.values)-1))
        out["faces"]["octagon"] = self.normalize_face(octv, channel="octagon", idx_range=(0, len(octv.values)-1))
        # Human summary
        summary = self.out / "summary.txt"
        with summary.open("w", encoding="utf-8") as f:
            f.write(f"Policy: {self.policy.name}\n")
            for ch in ("decagon","octagon"):
                s = out["faces"][ch]["state"]
                f.write(f"{ch}: θ={s['theta_deg']:.2f}°, W={s['W']}, repair={s['repair']}, K={s['clones_K']}\n")
        return out

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
