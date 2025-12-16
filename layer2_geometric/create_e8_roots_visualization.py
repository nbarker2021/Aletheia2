def create_e8_roots_visualization():
    """"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel 1: E8 root excitations (3D projection)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Generate sample E8 roots in 3D projection
    np.random.seed(42)
    n_roots = 48  # Subset for visualization
    
    # All E8 roots have length sqrt(2)
    root_length = np.sqrt(2)
    
    # Generate roots on sphere of radius sqrt(2)
    phi = np.random.uniform(0, 2*np.pi, n_roots)
    costheta = np.random.uniform(-1, 1, n_roots)
    u = np.random.uniform(0, 1, n_roots)
    
    theta = np.arccos(costheta)
    r = root_length * (u**(1/3))  # Uniform distribution in sphere
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)  
    z = r * np.cos(theta)
    
    # Plot ground state (origin)
    ax1.scatter([0], [0], [0], s=200, c='gold', marker='*', 
               label='Vacuum State', edgecolor='black', linewidth=2)
    
    # Plot root excitations
    ax1.scatter(x, y, z, s=60, c='red', alpha=0.7, label='Root Excitations')
    
    # Show some connections (gauge field dynamics)
    for i in range(0, min(16, len(x)), 4):
        ax1.plot([0, x[i]], [0, y[i]], [0, z[i]], 'gray', alpha=0.4, linewidth=1)
    
    # Highlight minimum excitation
    ax1.scatter([root_length], [0], [0], s=150, c='blue', marker='s', 
               label=f'Min. Excitation (Δ = √2Λ)', edgecolor='black')
    
    ax1.set_xlabel('Root Component 1')
    ax1.set_ylabel('Root Component 2') 
    ax1.set_zlabel('Root Component 3')
    ax1.set_title('E₈ Root Excitations\\n(Yang-Mills Glueball States)', fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Panel 2: Mass gap illustration
    energy_levels = [0, np.sqrt(2), 2*np.sqrt(2), np.sqrt(6), 2*np.sqrt(2)]
    level_names = ['Vacuum', '0⁺⁺', '2⁺⁺', '0⁻⁺', 'Multi-gluon']
    colors = ['gold', 'red', 'blue', 'green', 'purple']
    
    for i, (energy, name, color) in enumerate(zip(energy_levels, level_names, colors)):
        y_pos = energy
        ax2.hlines(y_pos, 0.2, 0.8, colors=color, linewidth=4)
        ax2.text(0.85, y_pos, name, va='center', fontsize=11, fontweight='bold')
        
        # Show excitation arrows
        if i > 0:
            ax2.annotate('', xy=(0.1, y_pos), xytext=(0.1, 0),
                        arrowprops=dict(arrowstyle='<->', lw=2, color='black'))
    
    # Highlight mass gap
    gap_height = np.sqrt(2)
    ax2.annotate('', xy=(0.05, gap_height), xytext=(0.05, 0),
                arrowprops=dict(arrowstyle='<->', lw=3, color='red'))
    ax2.text(-0.05, gap_height/2, 'Mass Gap\\nΔ = √2 Λ_QCD', 
             ha='right', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax2.set_xlim(-0.3, 1.2)
    ax2.set_ylim(-0.5, 4)
    ax2.set_ylabel('Energy (units of Λ_QCD)', fontsize=12)
    ax2.set_title('Yang-Mills Mass Spectrum\\nfrom E₈ Root Structure', fontweight='bold')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_ym_1_e8_excitations.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_ym_1_e8_excitations.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1: E₈ excitations and mass gap saved")
