def create_gauge_field_embedding():
    \"\"\"Create diagram showing gauge field to E8 embedding\"\"\"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Yang-Mills Gauge Field
    ax1.text(0.5, 0.85, 'Yang-Mills Theory', ha='center', fontsize=16, fontweight='bold')
    
    # Show field configuration
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    
    # Simulate gauge field (vector field)
    U = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    V = -np.cos(2*np.pi*X) * np.sin(2*np.pi*Y)
    
    ax1.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2], 
               alpha=0.7, scale=15, color='blue')
    
    ax1.text(0.5, 0.65, 'Gauge Field A_μ(x)', ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.text(0.5, 0.55, 'Gauss Law: D·E = 0', ha='center', fontsize=11)
    ax1.text(0.5, 0.45, 'Gauge Invariance', ha='center', fontsize=11)
    
    ax1.text(0.5, 0.25, 'Physical States:', ha='center', fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.18, '• Glueballs', ha='center', fontsize=10)
    ax1.text(0.5, 0.12, '• Bound states', ha='center', fontsize=10)
    ax1.text(0.5, 0.06, '• Mass gap Δ > 0 ??', ha='center', fontsize=10, color='red')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0.05, 0.02), 0.9, 0.96, fill=False, linewidth=2))
    
    # Panel 2: Cartan-Weyl Decomposition
    ax2.text(0.5, 0.9, 'Cartan-Weyl\\nDecomposition', ha='center', fontsize=16, fontweight='bold')
    
    ax2.text(0.5, 0.75, 'A_μ = Σᵢ aᵢ_μ Hᵢ + Σ_α a_α_μ E_α', ha='center', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    # Show 8 Cartan generators
    cartan_colors = plt.cm.Set3(np.linspace(0, 1, 8))
    for i in range(8):
        y_pos = 0.6 - i * 0.06
        ax2.add_patch(plt.Rectangle((0.1, y_pos-0.02), 0.8, 0.04, 
                                   facecolor=cartan_colors[i], alpha=0.7))
        ax2.text(0.05, y_pos, f'H₍{i+1}₎', ha='right', va='center', fontsize=10)
    
    ax2.text(0.5, 0.08, '8 Cartan Generators\\n+ 240 Root Generators', 
             ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Panel 3: E8 Lattice Structure
    ax3.text(0.5, 0.9, 'E₈ Lattice\\nEmbedding', ha='center', fontsize=16, fontweight='bold')
    
    # Show lattice points
    lattice_x = np.array([0.3, 0.7, 0.5, 0.4, 0.6, 0.35, 0.65])
    lattice_y = np.array([0.7, 0.7, 0.5, 0.6, 0.4, 0.45, 0.55])
    
    ax3.scatter(lattice_x, lattice_y, s=100, c='red', alpha=0.8, edgecolor='black')
    
    # Connect lattice points
    for i in range(len(lattice_x)-1):
        ax3.plot([lattice_x[i], lattice_x[i+1]], [lattice_y[i], lattice_y[i+1]], 
                'gray', alpha=0.5, linewidth=1)
    
    # Highlight center (vacuum)
    ax3.scatter([0.5], [0.5], s=200, c='gold', marker='*', 
               edgecolor='black', linewidth=2)
    ax3.text(0.52, 0.48, 'Vacuum', fontsize=10, fontweight='bold')
    
    # Show root excitations
    ax3.arrow(0.5, 0.5, 0.15, 0.15, head_width=0.03, head_length=0.02, 
             fc='blue', ec='blue', linewidth=2)
    ax3.text(0.68, 0.68, 'Root\\nExcitation', ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"))
    
    ax3.text(0.5, 0.25, 'Physical Constraint:', ha='center', fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.18, 'Configuration ∈ Λ₈', ha='center', fontsize=11)
    ax3.text(0.5, 0.11, 'Min. Energy = √2 Λ_QCD', ha='center', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax3.set_xlim(0.2, 0.8)
    ax3.set_ylim(0.2, 0.8)
    ax3.axis('off')
    
    # Add arrows between panels
    fig.text(0.31, 0.5, '→', fontsize=24, ha='center', va='center', fontweight='bold')
    fig.text(0.64, 0.5, '→', fontsize=24, ha='center', va='center', fontweight='bold')
    
    plt.suptitle('Yang-Mills to E₈ Embedding Process', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure_ym_2_embedding.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_ym_2_embedding.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2: Gauge field embedding saved")
