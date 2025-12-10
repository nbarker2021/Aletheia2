def create_chaos_transition_diagram():
    \"\"\"Create diagram showing laminar-turbulent transition\"\"\"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel 1: Lyapunov exponent vs Reynolds number
    Re = np.logspace(1, 3, 100)  # Reynolds numbers from 10 to 1000
    Re_critical = 240
    
    # Theoretical Lyapunov exponent
    lambda_theory = np.zeros_like(Re)
    for i, re in enumerate(Re):
        if re < Re_critical:
            lambda_theory[i] = -0.1 * (Re_critical - re) / Re_critical  # Negative (stable)
        else:
            lambda_theory[i] = 0.05 * (re - Re_critical) / Re_critical  # Positive (chaotic)
    
    # Add noise to simulate experimental data
    np.random.seed(42)
    lambda_observed = lambda_theory + 0.02 * np.random.randn(len(Re))
    
    ax1.semilogx(Re, lambda_theory, 'b-', linewidth=3, label='E₈ Theory', alpha=0.8)
    ax1.semilogx(Re, lambda_observed, 'ro', markersize=4, alpha=0.6, label='Simulated Data')
    
    # Mark critical point
    ax1.axvline(Re_critical, color='green', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Critical Re = {Re_critical}')
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    # Shade regions
    ax1.axvspan(10, Re_critical, alpha=0.2, color='blue', label='Laminar (λ < 0)')
    ax1.axvspan(Re_critical, 1000, alpha=0.2, color='red', label='Turbulent (λ > 0)')
    
    ax1.set_xlabel('Reynolds Number (Re)', fontsize=12)
    ax1.set_ylabel('Lyapunov Exponent (λ)', fontsize=12)
    ax1.set_title('Laminar-Turbulent Transition\\nfrom E₈ Overlay Dynamics', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.15, 0.2)
    
    # Panel 2: Energy spectrum comparison
    k = np.logspace(0, 2, 50)  # Wavenumbers
    
    # Kolmogorov spectrum
    k_kolm = k[10:40]  # Inertial range
    E_kolm = k_kolm**(-5/3)
    E_kolm = E_kolm / E_kolm[0]  # Normalize
    
    # E8 theoretical spectrum
    E_e8 = np.zeros_like(k)
    for i, ki in enumerate(k):
        if 2 <= ki <= 50:  # E8 inertial range
            E_e8[i] = ki**(-5/3) * np.exp(-ki/50)  # With E8 cutoff
        else:
            E_e8[i] = 0.01 * ki**(-2)  # Viscous/injection ranges
    
    E_e8 = E_e8 / np.max(E_e8)
    
    ax2.loglog(k_kolm, E_kolm, 'b-', linewidth=3, label='Kolmogorov k⁻⁵/³')
    ax2.loglog(k, E_e8, 'r--', linewidth=3, label='E₈ Theory', alpha=0.8)
    
    # Mark E8 characteristic scales
    k_e8_roots = [4, 16, 64]  # Characteristic root separations
    for k_root in k_e8_roots:
        ax2.axvline(k_root, color='green', linestyle=':', alpha=0.7)
    
    ax2.text(6, 0.3, 'E₈ Root\\nScales', ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Add -5/3 slope reference
    k_ref = np.array([5, 20])
    E_ref = 0.1 * k_ref**(-5/3)
    ax2.loglog(k_ref, E_ref, 'k--', alpha=0.5)
    ax2.text(8, 0.008, '-5/3', fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Wavenumber (k)', fontsize=12)
    ax2.set_ylabel('Energy Spectrum E(k)', fontsize=12)
    ax2.set_title('Turbulent Energy Spectrum\\nfrom E₈ Root Correlations', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0.001, 2)
    
    plt.tight_layout()
    plt.savefig('figure_ns_2_chaos_transition.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_ns_2_chaos_transition.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2: Chaos transition diagram saved")
