def create_overlay_flow_visualization():
    \"\"\"Create visualization of fluid parcels as E8 overlays\"\"\"
    fig = plt.figure(figsize=(16, 6))
    
    # Panel 1: Classical fluid view
    ax1 = plt.subplot(1, 3, 1)
    
    # Generate fluid parcel trajectories
    t = np.linspace(0, 4*np.pi, 100)
    n_parcels = 8
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_parcels))
    
    for i in range(n_parcels):
        # Spiral trajectories (streamlines)
        phase = 2*np.pi * i / n_parcels
        r = 0.8 + 0.2 * np.sin(t + phase)
        x = r * np.cos(t + phase)
        y = r * np.sin(t + phase)
        
        ax1.plot(x, y, color=colors[i], linewidth=2, alpha=0.8)
        
        # Mark initial positions
        ax1.scatter(x[0], y[0], color=colors[i], s=100, marker='o', 
                   edgecolor='black', linewidth=2, zorder=5)
        
        # Mark current positions  
        ax1.scatter(x[50], y[50], color=colors[i], s=80, marker='s',
                   edgecolor='black', linewidth=1, zorder=5)
    
    # Add velocity vectors
    theta = np.linspace(0, 2*np.pi, 12)
    x_vec = 0.6 * np.cos(theta)
    y_vec = 0.6 * np.sin(theta)
    u_vec = -0.3 * np.sin(theta)  # Tangential velocity
    v_vec = 0.3 * np.cos(theta)
    
    ax1.quiver(x_vec, y_vec, u_vec, v_vec, alpha=0.6, scale=5, color='red')
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Classical View:\\nFluid Parcels & Streamlines', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Panel 2: E8 overlay space
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    
    # Generate overlay positions (3D projection of 8D)
    np.random.seed(42)
    n_overlays = 20
    
    # Initial overlay configuration
    overlays_initial = []
    overlays_evolved = []
    
    for i in range(n_overlays):
        # Initial state
        r_init = 2 * (np.random.rand(8) - 0.5)  # Random in [-1, 1]^8
        
        # Evolved state (simulate MORSR dynamics)
        r_evolved = r_init + 0.3 * np.random.randn(8)  # Small perturbation
        
        overlays_initial.append(r_init)
        overlays_evolved.append(r_evolved)
    
    overlays_initial = np.array(overlays_initial)
    overlays_evolved = np.array(overlays_evolved)
    
    # Plot initial positions (3D projection)
    ax2.scatter(overlays_initial[:, 0], overlays_initial[:, 1], overlays_initial[:, 2],
               c='blue', s=60, alpha=0.8, label='Initial Overlays', edgecolor='black')
    
    # Plot evolved positions
    ax2.scatter(overlays_evolved[:, 0], overlays_evolved[:, 1], overlays_evolved[:, 2],
               c='red', s=60, alpha=0.8, label='Evolved Overlays', marker='s', edgecolor='black')
    
    # Draw evolution arrows
    for i in range(n_overlays):
        ax2.plot([overlays_initial[i, 0], overlays_evolved[i, 0]],
                [overlays_initial[i, 1], overlays_evolved[i, 1]], 
                [overlays_initial[i, 2], overlays_evolved[i, 2]], 
                'gray', alpha=0.5, linewidth=1)
    
    # Show E8 boundary (simplified as sphere)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = 2 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 2 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 2 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='green')
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_zlim(-2.5, 2.5)
    ax2.set_title('E₈ Overlay Space:\\n(3D Projection)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('E₈ Coord 1')
    ax2.set_ylabel('E₈ Coord 2')
    ax2.set_zlabel('E₈ Coord 3')
    ax2.legend(loc='upper right')
    
    # Panel 3: MORSR dynamics equations
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    # Display key equations
    equations = [
        "Navier-Stokes Equations:",
        r"$\\frac{\\partial \\mathbf{u}}{\\partial t} + (\\mathbf{u} \\cdot \\nabla)\\mathbf{u} = -\\nabla p + \\nu \\nabla^2 \\mathbf{u}$",
        r"$\\nabla \\cdot \\mathbf{u} = 0$",
        "",
        "↕ Equivalent to ↕",
        "",
        "MORSR Overlay Dynamics:",
        r"$\\frac{d\\mathbf{r}_i}{dt} = -\\frac{\\partial U}{\\partial \\mathbf{r}_i} + \\boldsymbol{\\eta}_i(t)$",
        r"$\\mathbf{r}_i \\in \\Lambda_8$ (E₈ lattice)",
        "",
        "Key Mappings:",
        "• Fluid parcels ↔ E₈ overlays",
        "• Velocity field ↔ Overlay motion", 
        "• Turbulence ↔ Chaotic dynamics",
        "• Viscosity ↔ Geometric damping"
    ]
    
    y_pos = 0.95
    for eq in equations:
        if eq.startswith(r"$") and eq.endswith(r"$"):
            # Mathematical equation
            ax3.text(0.1, y_pos, eq, fontsize=11, transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        elif eq.startswith("•"):
            # Bullet point
            ax3.text(0.15, y_pos, eq, fontsize=10, transform=ax3.transAxes)
        elif "↕" in eq:
            # Equivalence arrow
            ax3.text(0.5, y_pos, eq, fontsize=12, fontweight='bold', 
                    transform=ax3.transAxes, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        elif eq == "":
            # Skip blank lines (just decrement y)
            pass
        else:
            # Headers
            ax3.text(0.1, y_pos, eq, fontsize=12, fontweight='bold', 
                    transform=ax3.transAxes)
        
        y_pos -= 0.06
    
    ax3.set_title('Mathematical Framework', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure_ns_1_overlay_flow.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_ns_1_overlay_flow.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1: Overlay flow visualization saved")
