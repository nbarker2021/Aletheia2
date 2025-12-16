def create_sat_encoding_diagram():
    """"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: SAT Formula
    ax1.text(0.5, 0.8, 'SAT Formula φ', ha='center', fontsize=16, fontweight='bold')
    ax1.text(0.5, 0.65, 'Variables: x₁, x₂, ..., x₈', ha='center', fontsize=12)
    ax1.text(0.5, 0.55, 'Assignment: σ = (0,1,1,0,1,0,1,1)', ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax1.text(0.5, 0.4, 'Clauses:', ha='center', fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.32, 'C₁ = (x₁ ∨ ¬x₂ ∨ x₃)', ha='center', fontsize=10)
    ax1.text(0.5, 0.26, 'C₂ = (¬x₁ ∨ x₄ ∨ ¬x₅)', ha='center', fontsize=10)
    ax1.text(0.5, 0.2, '⋮', ha='center', fontsize=12)
    ax1.text(0.5, 0.14, 'Cₘ = (x₂ ∨ x₆ ∨ ¬x₈)', ha='center', fontsize=10)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, linewidth=2))
    
    # Panel 2: Encoding Process
    ax2.text(0.5, 0.8, 'E₈ Encoding', ha='center', fontsize=16, fontweight='bold')
    
    # Show 8 blocks
    block_colors = plt.cm.Set3(np.linspace(0, 1, 8))
    for i in range(8):
        y_pos = 0.65 - i * 0.07
        ax2.add_patch(plt.Rectangle((0.2, y_pos-0.02), 0.6, 0.04, 
                                   facecolor=block_colors[i], alpha=0.7))
        ax2.text(0.15, y_pos, f'h₍{i+1}₎', ha='right', va='center', fontsize=10)
        
        # Show variable assignments in block
        if i == 0:
            ax2.text(0.5, y_pos, 'x₁=0', ha='center', va='center', fontsize=8)
        elif i == 1:
            ax2.text(0.5, y_pos, 'x₂,x₃=1,1', ha='center', va='center', fontsize=8)
    
    ax2.text(0.5, 0.1, 'Point in Cartan Subalgebra', ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Panel 3: Weyl Chamber
    ax3.text(0.5, 0.9, 'Weyl Chamber', ha='center', fontsize=16, fontweight='bold')
    
    # Draw simplified chamber
    chamber_vertices = np.array([[0.3, 0.3], [0.7, 0.3], [0.6, 0.7], [0.4, 0.7]])
    chamber = Polygon(chamber_vertices, facecolor='lightgreen', alpha=0.5, 
                      edgecolor='green', linewidth=2)
    ax3.add_patch(chamber)
    
    # Mark point
    ax3.plot(0.5, 0.5, 'ro', markersize=10, label='Assignment Point')
    ax3.text(0.52, 0.52, 'p_σ', fontsize=12, fontweight='bold')
    
    # Show chamber boundaries
    ax3.text(0.25, 0.6, 'Root\\nHyperplane', ha='center', fontsize=8, rotation=45)
    ax3.plot([0.2, 0.8], [0.2, 0.8], 'k--', alpha=0.5)
    
    ax3.text(0.5, 0.15, 'Satisfying Assignment =\\nSpecific Chamber', 
             ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Add arrows
    ax1.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax2.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    plt.suptitle('SAT to E₈ Encoding Process', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure_3_sat_encoding.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_3_sat_encoding.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3: SAT encoding diagram saved")
