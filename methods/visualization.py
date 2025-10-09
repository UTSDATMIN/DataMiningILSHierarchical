import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from .clustering_manual import ward_distance, agglomerative_with_history

def figures_cluster_distribution(kuisoner_data):
    cluster_counts = kuisoner_data['Cluster'].value_counts()
    fig_pie = px.pie(
        values=cluster_counts.values,
        names=cluster_counts.index,
        title="Distribution of Students Across Clusters",
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    fig_bar = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        title="Number of Students per Cluster",
        labels={'x': 'Cluster', 'y': 'Number of Students'},
        color=cluster_counts.index,
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    return fig_pie, fig_bar

def figure_learning_style_subplots(kuisoner_data):
    dimensions = ['AR', 'SI', 'VV', 'QG']
    fig_styles = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Active vs Reflective', 'Sensing vs Intuitive',
                        'Visual vs Verbal', 'Sequential vs Global'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    for i, dim in enumerate(dimensions):
        counts = kuisoner_data[dim].value_counts()
        row, col = positions[i]
        fig_styles.add_trace(
            go.Bar(x=counts.index, y=counts.values, marker_color=colors[i], showlegend=False),
            row=row, col=col
        )
    fig_styles.update_layout(height=600, title_text="Learning Style Preferences Distribution")
    return fig_styles

def figure_dendrogram_with_library(X, ids):
    # matches your original call; local import avoids changing your global imports
    from scipy.cluster.hierarchy import linkage
    Z = linkage(X, method='ward', metric='euclidean')
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=list(ids), ax=ax, leaf_rotation=90)
    ax.set_title("Dendrogram of Learning Style Clusters")
    ax.set_xlabel("Student ID")
    ax.set_ylabel("Distance")
    return fig

def figure_custom_dendrogram(X):
    history = agglomerative_with_history(X)
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_positions = {}
    for idx in range(len(X)):
        cluster_positions[frozenset([idx])] = (idx, 0)
        ax.text(idx, -0.2, str(idx), ha='center')
    for c1, c2, dist in history:
        pos1 = cluster_positions[frozenset(c1)]
        pos2 = cluster_positions[frozenset(c2)]
        new_x = (pos1[0] + pos2[0]) / 2
        new_y = dist
        ax.plot([pos1[0], pos1[0]], [pos1[1], new_y], c='k')
        ax.plot([pos2[0], pos2[0]], [pos2[1], new_y], c='k')
        ax.plot([pos1[0], pos2[0]], [new_y, new_y], c='k')
        cluster_positions[frozenset(c1+c2)] = (new_x, new_y)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Distance")
    ax.set_title("Custom Dendrogram (Ward’s Method)")
    return fig
