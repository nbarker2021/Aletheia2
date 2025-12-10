def ensemble_metrics(ensemble, Rstar, per_pack):
    best_rates = []; ticket_rates = []; disc_ticket_rates = []
    for name, st in per_pack.items():
        Rset = fixed_rotations(2025)
        br = -1.0
        for R in Rset:
            P = pose_bits(st["X8"], st["V"], E8_ROOTS, R); r,_ = alignment_rate(P)
            if r>br: br=r
        best_rates.append(br)
        tickets = (st["margins"] <= 0.05)
        ticket_rates.append(float(tickets.mean()))
        Pstar = pose_bits(st["X8"], st["V"], E8_ROOTS, Rstar)
        rstar, ints = alignment_rate(Pstar)
        vals, counts = np.unique(ints, return_counts=True)
        modal = vals[np.argmax(counts)]
        disc = (ints != modal)
        disc_ticket_rates.append(float((tickets & disc).mean()))
    ensemble_pose_rate = float(np.mean([alignment_rate(pose_bits(st["X8"], st["V"], E8_ROOTS, Rstar))[0] for st in per_pack.values()]))
    mean_best_rate = float(np.mean(best_rates))
    pose_loss = mean_best_rate - ensemble_pose_rate
    ticket_rate = float(np.mean(ticket_rates))
    disc_ticket_rate = float(np.mean(disc_ticket_rates))
    synergy = ensemble_pose_rate * (1.0 - disc_ticket_rate)
    return {
        "ensemble_pose_rate": ensemble_pose_rate,
        "mean_best_rate": mean_best_rate,
        "pose_loss": pose_loss,
        "ticket_rate": ticket_rate,
        "discordant_ticket_rate": disc_ticket_rate,
        "synergy_index": synergy
    }

# -------- Overlays --------