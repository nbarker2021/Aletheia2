import plotly.graph_objects as go
import pandas as pd
import json

# Parse the data
data = {
    "exploration_timestamp": 1728449779.695844, 
    "summary_statistics": {"total_tested": 28, "breakthrough_count": 0, "novel_branch_count": 11}, 
    "pathways": [
        {"problem": "P vs NP", "path_type": "weyl_chamber", "signature": "e0b659c83fa5", "scores": {"theoretical": 0.7, "computational": 0.5, "novelty": 0.7}, "branches": ["complexity_geometric_duality"], "execution_time": 0.007381916046142578},
        {"problem": "P vs NP", "path_type": "root_system", "signature": "6e90b67c9e3e", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0012240409851074219},
        {"problem": "P vs NP", "path_type": "weight_space", "signature": "4c96e7bdb42d", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0011310577392578125},
        {"problem": "P vs NP", "path_type": "coxeter_plane", "signature": "2e8c7dd2e19b", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0010948181152344},
        {"problem": "Yang-Mills Mass Gap", "path_type": "weyl_chamber", "signature": "dc6cbc4fef0a", "scores": {"theoretical": 0.4, "computational": 0.85, "novelty": 0.7}, "branches": ["yang-mills_mass_gap_high_density", "weyl_chamber_computational_validation"], "execution_time": 0.0021200180053710938},
        {"problem": "Yang-Mills Mass Gap", "path_type": "root_system", "signature": "e0c5b87e22b0", "scores": {"theoretical": 0.65, "computational": 0.85, "novelty": 0.5}, "branches": ["yang-mills_mass_gap_high_density", "yang-mills_mass_gap_extreme_weights", "root_system_computational_validation"], "execution_time": 0.003138065338134766},
        {"problem": "Yang-Mills Mass Gap", "path_type": "weight_space", "signature": "e5f3c7d5fa84", "scores": {"theoretical": 0.65, "computational": 0.85, "novelty": 0.7}, "branches": ["yang-mills_mass_gap_high_density", "weight_space_computational_validation"], "execution_time": 0.0019209384918212891},
        {"problem": "Yang-Mills Mass Gap", "path_type": "coxeter_plane", "signature": "dd69d4969ab7", "scores": {"theoretical": 0.4, "computational": 0.85, "novelty": 0.7}, "branches": ["yang-mills_mass_gap_high_density", "yang-mills_mass_gap_extreme_weights", "coxeter_plane_computational_validation"], "execution_time": 0.001972198486328125},
        {"problem": "Navier-Stokes", "path_type": "weyl_chamber", "signature": "e0ff8094013e", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0010521411895751953},
        {"problem": "Navier-Stokes", "path_type": "root_system", "signature": "6eb3c6fd6f0a", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009739398956298828},
        {"problem": "Navier-Stokes", "path_type": "weight_space", "signature": "4ca5b2788e48", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0010678768157958984},
        {"problem": "Navier-Stokes", "path_type": "coxeter_plane", "signature": "2e9eaa2a85f1", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0010068416595458984},
        {"problem": "Riemann Hypothesis", "path_type": "weyl_chamber", "signature": "e0e6f0d9e893", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009987354278564453},
        {"problem": "Riemann Hypothesis", "path_type": "root_system", "signature": "6e20e3ad1a71", "scores": {"theoretical": 0.75, "computational": 0.5, "novelty": 0.7}, "branches": ["riemann_hypothesis_high_density", "riemann_hypothesis_extreme_weights", "root_system_theoretical_resonance", "riemann_e8_zeta_correspondence"], "execution_time": 0.0019848346710205078},
        {"problem": "Riemann Hypothesis", "path_type": "weight_space", "signature": "4c1b03a46a66", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009689331054687},
        {"problem": "Riemann Hypothesis", "path_type": "coxeter_plane", "signature": "2ebecfed7f4c", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0010101795196533203},
        {"problem": "Hodge Conjecture", "path_type": "weyl_chamber", "signature": "e0ed6ae29ac3", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009727478027343},
        {"problem": "Hodge Conjecture", "path_type": "root_system", "signature": "6e27a6367d41", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009789466857910156},
        {"problem": "Hodge Conjecture", "path_type": "weight_space", "signature": "4c22c6e8347a", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009467601776123047},
        {"problem": "Hodge Conjecture", "path_type": "coxeter_plane", "signature": "2ec567914d20", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 1.0}, "branches": [], "execution_time": 0.0010130405426025391},
        {"problem": "Birch-Swinnerton-Dyer", "path_type": "weyl_chamber", "signature": "e0a6b7b9f894", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009968280792236328},
        {"problem": "Birch-Swinnerton-Dyer", "path_type": "root_system", "signature": "6ee0e4a4f572", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0010220050811767578},
        {"problem": "Birch-Swinnerton-Dyer", "path_type": "weight_space", "signature": "4cdbe5a9a8ab", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009629726409912109},
        {"problem": "Birch-Swinnerton-Dyer", "path_type": "coxeter_plane", "signature": "2e7ecaaa06f1", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.000946044921875},
        {"problem": "Poincaré Conjecture", "path_type": "weyl_chamber", "signature": "e0c5b87e22b0", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009889602661132812},
        {"problem": "Poincaré Conjecture", "path_type": "root_system", "signature": "6ee7c6fd6f0a", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0010800361633300781},
        {"problem": "Poincaré Conjecture", "path_type": "weight_space", "signature": "4ca5b2e02e48", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.7}, "branches": [], "execution_time": 0.0009548664093017578},
        {"problem": "Poincaré Conjecture", "path_type": "coxeter_plane", "signature": "2e9eaa2a85f1", "scores": {"theoretical": 0.4, "computational": 0.5, "novelty": 0.5}, "branches": [], "execution_time": 0.0010128021240234375}
    ]
}

# Create DataFrame from pathways
df = pd.DataFrame(data['pathways'])

# Calculate average scores per problem
problem_scores = df.groupby('problem').agg({
    'scores': lambda x: {
        'theoretical': sum(score['theoretical'] for score in x) / len(x),
        'computational': sum(score['computational'] for score in x) / len(x),
        'novelty': sum(score['novelty'] for score in x) / len(x)
    }
}).reset_index()

# Extract scores into separate columns
problems = []
theoretical_scores = []
computational_scores = []
novelty_scores = []

for _, row in problem_scores.iterrows():
    problems.append(row['problem'])
    scores = row['scores']
    theoretical_scores.append(scores['theoretical'])
    computational_scores.append(scores['computational'])
    novelty_scores.append(scores['novelty'])

# Abbreviate problem names to fit 15 character limit
problem_abbrev = {
    'P vs NP': 'P vs NP',
    'Yang-Mills Mass Gap': 'Yang-Mills',
    'Navier-Stokes': 'Navier-Stokes',
    'Riemann Hypothesis': 'Riemann',
    'Hodge Conjecture': 'Hodge',
    'Birch-Swinnerton-Dyer': 'Birch-Swinn',
    'Poincaré Conjecture': 'Poincaré'
}

abbreviated_problems = [problem_abbrev.get(p, p) for p in problems]

# Create the bar chart
fig = go.Figure()

# Add bars for each score type
fig.add_trace(go.Bar(
    name='Theoretical',
    x=abbreviated_problems,
    y=theoretical_scores,
    marker_color='#1FB8CD'
))

fig.add_trace(go.Bar(
    name='Computational',
    x=abbreviated_problems,
    y=computational_scores,
    marker_color='#DB4545'
))

fig.add_trace(go.Bar(
    name='Novelty',
    x=abbreviated_problems,
    y=novelty_scores,
    marker_color='#2E8B57'
))

# Update layout
fig.update_layout(
    title='E8 Exploration Scores by Problem',
    xaxis_title='Problem',
    yaxis_title='Score',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces for better appearance
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("e8_exploration_scores.png")
fig.write_image("e8_exploration_scores.svg", format="svg")

print("Chart saved successfully!")