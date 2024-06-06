from IPython.display import HTML, display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from pathlib import Path

class Generate_Report:
    def __init__(self):
        pass

    def plt_img_base64(self, img_base64):
        """
        Display base64 encoded string as image.

        :param img_base64: Base64 string
        """
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        display(HTML(image_html))

    def generate_image_base64(self, fig):
        """
        Convert a matplotlib figure to a base64 encoded string.

        :param fig: Matplotlib figure
        :return: Base64 string
        """
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return img_base64

    def generate_report(self, df, knowledge_graph_path, context_proximity_path):
        """
        Generate a comprehensive report from the given DataFrame and paths.

        :param df: DataFrame containing the main data
        :param knowledge_graph_path: Path to the knowledge graph CSV file
        :param context_proximity_path: Path to the context proximity CSV file
        """
        # Report Title
        display(HTML("<h1>Data Generation and Processing Report</h1>"))

        # Display DataFrame information
        display(HTML("<h2>Data Overview</h2>"))
        display(HTML(df.to_html()))
        display(HTML("<h3>Data Description</h3>"))
        display(HTML(df.describe().to_html()))

        # Load and display knowledge graph data
        knowledge_graph_df = pd.read_csv(knowledge_graph_path, sep=";")
        display(HTML("<h2>Knowledge Graph Data</h2>"))
        display(HTML(knowledge_graph_df.to_html()))
        self.plot_knowledge_graph(knowledge_graph_df)

        # Load and display context proximity data
        context_proximity_df = pd.read_csv(context_proximity_path, sep=";")
        display(HTML("<h2>Context Proximity Data</h2>"))
        display(HTML(context_proximity_df.to_html()))
        self.plot_context_proximity(context_proximity_df)

    def plot_knowledge_graph(self, df):
        """
        Plot and display the knowledge graph.

        :param df: DataFrame containing the knowledge graph data
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='node_1')
        plt.title('Knowledge Graph Node Frequency')
        plt.xlabel('Node')
        plt.ylabel('Count')

        # Convert plot to base64 and display
        img_base64 = self.generate_image_base64(plt.gcf())
        self.plt_img_base64(img_base64)
        plt.close()

    def plot_context_proximity(self, df):
        """
        Plot and display the context proximity data.

        :param df: DataFrame containing the context proximity data
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.pivot("node_1", "node_2", "count"), cmap="YlGnBu", annot=True)
        plt.title('Context Proximity Heatmap')
        plt.xlabel('Node 1')
        plt.ylabel('Node 2')

        # Convert plot to base64 and display
        img_base64 = self.generate_image_base64(plt.gcf())
        self.plt_img_base64(img_base64)
        plt.close()



    # def generate_report(self, df, knowledge_graph_path, context_proximity_path):
    #     """
    #     Generate a comprehensive terminal report from the given DataFrame and paths.

    #     :param df: DataFrame containing the main data
    #     :param knowledge_graph_path: Path to the knowledge graph CSV file
    #     :param context_proximity_path: Path to the context proximity CSV file
    #     """
    #     # Report Title
    #     self.print_report_title()

    #     # Display DataFrame information
    #     self.display_data_overview(df)

    #     # Load and display knowledge graph data
    #     knowledge_graph_df = pd.read_csv(knowledge_graph_path, sep=";")
    #     self.display_knowledge_graph_data(knowledge_graph_df)

    #     # Load and display context proximity data
    #     context_proximity_df = pd.read_csv(context_proximity_path, sep=";")
    #     self.display_context_proximity_data(context_proximity_df)

    #     # Additional insights
    #     self.additional_insights(df, knowledge_graph_df, context_proximity_df)

    # def print_report_title(self):
    #     """Prints the report title."""
    #     print("\nData Generation and Processing Report\n")
    #     print("=" * 40)

    # def display_data_overview(self, df):
    #     """Displays an overview of the main data."""
    #     print("\nData Overview\n")
    #     print(df.info())
    #     print("\nFirst few rows of the data:\n")
    #     print(df.head())
    #     print("\nData Description:\n")
    #     print(df.describe())

    # def display_knowledge_graph_data(self, knowledge_graph_df):
    #     """Displays an overview of the knowledge graph data."""
    #     print("\nKnowledge Graph Data\n")
    #     print("=" * 40)
    #     print("\nFirst few rows of the knowledge graph data:\n")
    #     print(knowledge_graph_df.head())
    #     print("\nKnowledge Graph Data Statistics:\n")
    #     print(knowledge_graph_df.describe())

    # def display_context_proximity_data(self, context_proximity_df):
    #     """Displays an overview of the context proximity data."""
    #     print("\nContext Proximity Data\n")
    #     print("=" * 40)
    #     print("\nFirst few rows of the context proximity data:\n")
    #     print(context_proximity_df.head())
    #     print("\nContext Proximity Data Statistics:\n")
    #     print(context_proximity_df.describe())

    # def additional_insights(self, df, knowledge_graph_df, context_proximity_df):
    #     """
    #     Print additional insights from the data.

    #     :param df: Main data DataFrame
    #     :param knowledge_graph_df: Knowledge graph data DataFrame
    #     :param context_proximity_df: Context proximity data DataFrame
    #     """
    #     print("\nAdditional Insights\n")
    #     print("=" * 40)

    #     # Main data insights
    #     self.print_main_data_insights(df)

    #     # Knowledge graph insights
    #     self.print_knowledge_graph_insights(knowledge_graph_df)

    #     # Context proximity insights
    #     self.print_context_proximity_insights(context_proximity_df)

    # def print_main_data_insights(self, df):
    #     """Prints insights from the main data."""
    #     print("\nMain Data Insights:\n")
    #     print(f"Total records: {len(df)}")
    #     print(f"Number of unique chunks: {df['chunk_id'].nunique()}")
    #     print(f"Text length distribution:\n{df['text'].apply(len).describe()}")

    # def print_knowledge_graph_insights(self, knowledge_graph_df):
    #     """Prints insights from the knowledge graph data."""
    #     print("\nKnowledge Graph Insights:\n")
    #     print(f"Total records: {len(knowledge_graph_df)}")
    #     print(f"Number of unique nodes: {pd.concat([knowledge_graph_df['node_1'], knowledge_graph_df['node_2']]).nunique()}")
    #     print(f"Number of unique edges: {knowledge_graph_df['edge'].nunique()}")

    # def print_context_proximity_insights(self, context_proximity_df):
    #     """Prints insights from the context proximity data."""
    #     print("\nContext Proximity Insights:\n")
    #     print(f"Total records: {len(context_proximity_df)}")
    #     print(f"Number of unique nodes: {pd.concat([context_proximity_df['node_1'], context_proximity_df['node_2']]).nunique()}")
    #     print(f"Context proximity count distribution:\n{context_proximity_df['count'].describe()}")

