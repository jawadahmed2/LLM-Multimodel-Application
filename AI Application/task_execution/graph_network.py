import networkx as nx
import seaborn as sns
from pyvis.network import Network
import pandas as pd
import random

def create_graph(nodes, dfg):
    """
    Create a graph based on nodes and edges.

    Args:
    - nodes (list): List of nodes for the graph.
    - dfg (pd.DataFrame): DataFrame containing edges information.

    Returns:
    - G (nx.Graph): Generated graph.
    - communities (list): List of communities in the graph.
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )

    # Add edges to the graph
    for index, row in dfg.iterrows():
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row['count']/4
        )

    # Calculate communities for coloring the nodes
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    print("Number of Communities = ", len(communities))
    print(communities)

    return G, communities


def colors2Community(communities) -> pd.DataFrame:
    """
    Convert communities into colors for visualization.

    Args:
    - communities (list): List of communities.

    Returns:
    - df_colors (pd.DataFrame): DataFrame containing node colors and groups.
    """
    palette = "hls"
    # Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors


def display_graph(G):
    """
    Display the graph using Pyvis.

    Args:
    - G (nx.Graph): Graph to be displayed.
    """
    net = Network(
        notebook=True,
        cdn_resources="remote",
        height="800px",
        width="100%",
        select_menu=True,
        filter_menu=False,
    )

    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)

    net.show_buttons(filter_=['physics'])
    net.show("data_preparation/data/KGraph_data/knowledge_graph.html")
