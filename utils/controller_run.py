def controller_run(X, outdir, cycles=4, tau_w=0.05, tau_annih=0.01, seed=2025, packs_json=None, ensemble_auto=False):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Normalize
    Xn = X.copy().astype(float)
    colmax = np.maximum(1.0, np.max(np.abs(Xn), axis=0))
    Xn = Xn / colmax

    ensembles_rows = []; tickets_rows = []; cycles_rows = []
    prev_ticket_count = None; discovery_stalls = 0
    Rset = fixed_rotations(seed)

    def tickets_from_matrix(Z):
        blocks = block8s(Z)
        mlist = []
        for B in blocks:
            V, d_best, di, dh, coset, altV = e8_snap_block(B)
            m = coset_margin(di, dh); mlist.append(m)
        front = mlist[0] <= tau_w
        for k in range(1, len(mlist)):
            front |= (mlist[k] <= tau_w)
        return front

    for c in range(1, cycles+1):
        blocks = block8s(Xn)
        ensemble = {"MAIN_B0": blocks[0]}
        if len(blocks) > 1: ensemble["MAIN_B1"] = blocks[1]
        Rstar, mean_rate, per_pack, _ = ensemble_choose_Rstar(ensemble, Rset)
        em = ensemble_metrics(ensemble, Rstar, per_pack)
        ensembles_rows.append({"cycle": c, **em})

        # Snap blocks
        snap = []; margins = []
        for B in blocks:
            V, d_best, di, dh, coset, altV = e8_snap_block(B)
            m = coset_margin(di, dh); margins.append(m); snap.append((B,V,di,dh,coset,altV))

        # Tickets
        front = margins[0] <= tau_w
        for k in range(1, len(margins)): front |= (margins[k] <= tau_w)
        idxs = np.where(front)[0]
        cycles_rows.append({"cycle": c, "tickets": int(len(idxs))})

        # Overlays on first block
        X8, V8, di8, dh8, cos8, alt8 = snap[0]
        pd.DataFrame({"index": np.arange(len(X8)), "hnf": overlay_hnf(X8, V8, Rstar).astype(int)}).to_csv(Path(outdir)/"overlays_hnf.csv", index=False)
        pd.DataFrame({"index": np.arange(len(X8)), "dsc": overlay_dsc(X8, V8, Rstar).astype(int)}).to_csv(Path(outdir)/"overlays_dsc.csv", index=False)
        # PI
        pi = overlay_pi(Xn, tickets_from_matrix)
        Path(outdir/"overlays_pi.json").write_text(json.dumps(pi, indent=2))
        # Miners
        pose_spectrum(X8, V8, Rstar).to_csv(Path(outdir)/"pose_spectrum.csv", index=False)
        orbit_hash(X8, V8, Rstar).to_csv(Path(outdir)/"orbit_hash.csv", index=False)

        # Ticket rows
        if len(blocks)==1:
            Vcat = snap[0][1]; Altcat = snap[0][5]
        else:
            Vcat = np.hstack([s[1] for s in snap]); Altcat = np.hstack([s[5] for s in snap])
        move_cost = np.linalg.norm(Altcat - Vcat, axis=1)
        for i in idxs:
            margin_min = float(min([margins[k][i] for k in range(len(blocks))]))
            action = "hold"
            if margin_min <= 0.01: action = "annihilate_to_rails"
            elif move_cost[i] < 0.75: action = "consider_parity_flip"
            tickets_rows.append({"cycle": c, "index": int(i), "margin_min": margin_min,
                                 "move_cost": float(move_cost[i]), "proposed_action": action})

        if prev_ticket_count is not None and len(idxs) == prev_ticket_count:
            discovery_stalls += 1
        else:
            discovery_stalls = 0
        prev_ticket_count = len(idxs)
        if discovery_stalls >= 2: break

    # Write artifacts
    Path(outdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ensembles_rows).to_csv(Path(outdir)/"ensembles.csv", index=False)
    pd.DataFrame(cycles_rows).to_csv(Path(outdir)/"cycles.csv", index=False)
    if len(tickets_rows)>0: pd.DataFrame(tickets_rows).to_csv(Path(outdir)/"tickets.csv", index=False)
    summary = {
        "cycles_run": int(pd.DataFrame(cycles_rows)["cycle"].max()) if len(cycles_rows)>0 else 0,
        "last_ticket_count": int(pd.DataFrame(cycles_rows)["tickets"].iloc[-1]) if len(cycles_rows)>0 else 0,
        "saturated": bool(discovery_stalls >= 2)
    }
    Path(outdir/"summary.json").write_text(json.dumps(summary, indent=2))
    return summary
