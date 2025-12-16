import plotly.graph_objects as go
import plotly.express as px

# Extract data from the JSON
claims_data = [
    {"claim_id": "RIEMANN_E8_001", "validation_score": 0.4, "claim_status": "MODERATE_EVIDENCE"},
    {"claim_id": "RIEMANN_E8_002", "validation_score": 0.49166666666666664, "claim_status": "MODERATE_EVIDENCE"},
    {"claim_id": "COMPLEXITY_E8_001", "validation_score": 1.0, "claim_status": "STRONG_EVIDENCE"},
    {"claim_id": "COMPLEXITY_E8_002", "validation_score": 0.0006666666666666666, "claim_status": "INSUFFICIENT_EVIDENCE"}
]

# Create claim IDs within 15 char limit
claim_ids = ["RIEMANN_E8_001", "RIEMANN_E8_002", "COMPLX_E8_001", "COMPLX_E8_002"]
validation_scores = [round(claim["validation_score"], 3) for claim in claims_data]
evidence_levels = [claim["claim_status"] for claim in claims_data]

# Define colors based on evidence levels (following instructions: Strong=green, Moderate=yellow, Insufficient=red)
color_map = {
    "STRONG_EVIDENCE": "#2E8B57",  # Sea green
    "MODERATE_EVIDENCE": "#D2BA4C",  # Moderate yellow  
    "INSUFFICIENT_EVIDENCE": "#DB4545"  # Bright red
}

colors = [color_map[level] for level in evidence_levels]

# Create evidence level labels for legend
evidence_labels = []
for level in evidence_levels:
    if level == "STRONG_EVIDENCE":
        evidence_labels.append("Strong")
    elif level == "MODERATE_EVIDENCE":
        evidence_labels.append("Moderate")
    else:
        evidence_labels.append("Insufficient")

# Create the bar chart with separate traces for legend
fig = go.Figure()

# Add bars grouped by evidence level for proper legend
evidence_types = list(set(evidence_levels))
legend_added = set()

for i, (claim_id, score, level, color) in enumerate(zip(claim_ids, validation_scores, evidence_levels, colors)):
    # Determine legend label
    legend_label = evidence_labels[i]
    show_legend = legend_label not in legend_added
    
    if show_legend:
        legend_added.add(legend_label)
    
    fig.add_trace(go.Bar(
        x=[claim_id],
        y=[score],
        marker_color=color,
        text=[f"{score:.3f}"],
        textposition='outside',
        textfont=dict(size=12),
        name=legend_label,
        showlegend=show_legend
    ))

# Update layout
fig.update_layout(
    title="AI Math Claims Validation Scores",
    xaxis_title="Claim ID",
    yaxis_title="Valid Score",
    yaxis=dict(range=[0, max(validation_scores) * 1.2]),
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save both PNG and SVG
fig.write_image("validation_chart.png")
fig.write_image("validation_chart.svg", format="svg")

print("Chart saved successfully as validation_chart.png and validation_chart.svg")