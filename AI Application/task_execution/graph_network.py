import networkx as nx
import seaborn as sns
from pyvis.network import Network
import pandas as pd
import random

def create_graph(nodes, dfg):
    ## Create a graph
    G = nx.Graph()

    ## Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )

    ## Add edges to the graph
    for index, row in dfg.iterrows():
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row['count']/4
        )


    # ### Calculate communities for coloring the nodes

    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    print("Number of Communities = ", len(communities))
    print(communities)

    return G, communities


# ### Create a dataframe for community colors
## Now add these colors to communities and make another dataframe
def colors2Community(communities) -> pd.DataFrame:
    palette = "hls"
    ## Define a color palette
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

    net = Network(
        notebook=True,
        # bgcolor="#1a1a1a",
        cdn_resources="remote",
        height="800px",
        width="100%",
        select_menu=True,
        # font_color="#cccccc",
        filter_menu=False,
    )

    net.from_nx(G)
    # net.repulsion(node_distance=150, spring_length=400)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    # net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)

    net.show_buttons(filter_=['physics'])
    net.show("data_preparation/data/KGraph_data/knowledge_graph.html")

