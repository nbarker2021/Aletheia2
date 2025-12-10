def create_e8_projection_figure():
    \"\"\"Create 2D projection of E8 root system\"\"\"
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Generate sample E8 roots (240 total, show subset)
    np.random.seed(42)
    n_roots = 60  # Subset for visualization
    
    # E8 roots have norm sqrt(2), project to 2D
    angles = np.linspace(0, 2*np.pi, n_roots, endpoint=False)
    radius = np.sqrt(2)
    
    x = radius * np.cos(angles) + 0.1 * np.random.randn(n_roots)
    y = radius * np.sin(angles) + 0.1 * np.random.randn(n_roots)
    
    # Plot roots
    ax.scatter(x, y, s=50, alpha=0.7, c='red', label='E₈ Roots')
    
    # Show lattice structure with connecting lines
    for i in range(0, n_roots, 8):
        if i+8 < n_roots:
            ax.plot([x[i], x[i+8]], [y[i], y[i+8]], 'gray', alpha=0.3, linewidth=0.5)
    
    # Highlight special roots (simple roots)
    special_indices = [0, 8, 16, 24, 32, 40, 48, 56]
    ax.scatter(x[special_indices], y[special_indices], s=100, c='blue', 
               marker='s', label='Simple Roots', edgecolor='black', linewidth=1)
    
    # Add Weyl chamber boundaries (approximate)
    theta = np.linspace(0, 2*np.pi/8, 100)
    chamber_x = 2.5 * np.cos(theta)
    chamber_y = 2.5 * np.sin(theta)
    ax.plot(chamber_x, chamber_y, 'green', linewidth=3, label='Weyl Chamber')
    
    ax.fill_between(chamber_x, chamber_y, alpha=0.1, color='green')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('Cartan Coordinate 1', fontsize=12)
    ax.set_ylabel('Cartan Coordinate 2', fontsize=12)
    ax.set_title('E₈ Root System (2D Projection)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_1_e8_roots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_1_e8_roots.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1: E₈ root system saved")
