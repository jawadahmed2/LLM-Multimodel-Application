from client.llm_connection import LLMConnection
from data_preparation.data_generation import Data_Generation
from data_preparation.data_processing import Data_Processing
from data_preparation.prompt_template import Prompt_Template
from .rag_chats import GraphState, RagProcess
from .graph_network import create_graph, colors2Community, display_graph
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.chains import TransformChain
from langchain_core.runnables import chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain.retrievers import EnsembleRetriever
import pprint
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from loguru import logger

llm_connection = LLMConnection()
data_generation = Data_Generation()
data_processing = Data_Processing()
prompt_template = Prompt_Template()
rag_process = RagProcess()


class Task_Execution:
    """
    Class to execute various tasks using different processes and models.
    """

    def __init__(self):
        self.ollama = llm_connection.connect_ollama()
        self.chat_ollama = llm_connection.connect_chat_ollama()
        self.ollama_client, self.client_model = llm_connection.ollama_client()

    def execute_automate_browsing(self, search_query, user_prompt):
        """
        Execute automated browsing task using React model.
        """
        prompt = data_generation.get_react_prompting()
        serper_wrapper = GoogleSerperAPIWrapper()
        wikipedia_wrapper = WikipediaAPIWrapper()
        tools = [
            Tool(
                name="Intermediate Answer",
                description=user_prompt,
                func=serper_wrapper.run,
            ),
            Tool(
                name="Final Answer",
                description=user_prompt,
                func=wikipedia_wrapper.run,
            ),
        ]
        agent = create_react_agent(self.chat_ollama, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        output = agent_executor.invoke({"input": search_query})
        return output


    @staticmethod
    @chain
    def image_model(inputs: dict) -> str | list[str] | dict:
        """
        Execute model with image and prompt.
        """
        logger.info(f"Image Processing...")
        image = data_processing.image_processing(inputs)["image"]
        multi_model = llm_connection.connect_multimodel_ollama()
        multi_model.bind(images=image)
        msg = multi_model.invoke(
            [
                {"role": "system", "content": "you are a useful assistant that provides information about images"},
                {"role": "user", "content": inputs["prompt"]},
            ]
        )
        return msg

    def get_image_informations(self, image_path: str, user_prompt) -> dict:
        """
        Retrieve information about an image.
        """
        load_image_chain = TransformChain(
        input_variables=['image_path'],
        output_variables=['image'],
        transform=data_processing.image_processing
        )
        image_model = self.image_model
        vision_chain = load_image_chain | image_model
        output = vision_chain.invoke({'image_path': f'{image_path}', 'prompt': prompt_template.get_image_info_prompt(user_prompt)})
        return output

    def format_docs(self, docs):
        """
        Format documents.
        """
        return "\n\n".join([d.page_content for d in docs])

    def generate_instructions_training_data(self, is_gen_instruct=False, is_gen_training=False):
        """
        Generate instructions and training data.
        """
        print("Generating instructions and training data...")
        QA_PROMPT = prompt_template.llm_tunning_template()
        db, bm25_r = data_generation.load_db()

        output_parser = StrOutputParser()

        if is_gen_instruct:
            query = """
                Please generate two questions about SteelHead based on the provided context. The question should be around SteelHead WAN acceleration and its related concepts only. The questions should start with any of the following: "What", "How', "Is there a", "What are the", "How do I", "When is it", "Does SteelHead have", "How to", "What is the difference", "Which", "List". You do not need to provide an answer or category to each question.
                """

            # Custom QA Chain
            chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | QA_PROMPT
            | self.ollama
            | output_parser
        )
            data_processing.process_instructions(db, chain, query)

        if is_gen_training:
            faiss_retriever = db.as_retriever(
                search_type="mmr", search_kwargs={"fetch_k": 3}, max_tokens_limit=1000
            )
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_r, faiss_retriever], weights=[0.3, 0.7]
            )

            # Custom QA Chain
            chain = (
                {"context": ensemble_retriever | self.format_docs, "question": RunnablePassthrough()}
                | QA_PROMPT
                | self.ollama
                | output_parser
            )
            data_processing.process_training(db, bm25_r, chain)


    def execute_graph_prompt(self, input: str, model: str, metadata={}):
        """
        Execute graph prompt task.
        """
        print("Executing graph prompt...")
        chunk_id = metadata.get('chunk_id', None)
        USER_PROMPT, SYS_PROMPT = prompt_template.graphPrompt(input, chunk_id)
        response = self.ollama_client.generate(model, system=SYS_PROMPT, prompt=USER_PROMPT)
        aux1 = response['response']
        start_index = aux1.find('[')
        json_string = aux1[start_index:]
        json_string = json_string.replace("\\\\\_", "_")
        json_string = json_string.replace('\\\\_', '_')
        json_string = json_string.replace('\\\_', '_')
        json_string = json_string.replace('\\_', '_')
        json_string = json_string.replace('\_', '_')
        json_string.lstrip()
        print("Response JSON string:\n" + json_string)
        try:
            result = json.loads(json_string)
            result = [dict(item) for item in result]
        except:
            print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
            result = None
        return result


    def execute_knowledge_graph(self, regenerate_data):
        """
        Execute knowledge graph generation task.
        """
        print("Executing knowledge graph generation...")
        input_file_name = "Saxony_Eastern_Expansion_EP_96.txt"
        outputdirectory = Path("data_preparation/data/KGraph_data/data_output")

        output_graph_file_name = f"graph_{input_file_name[:-4]}.csv"
        output_graph_file_with_path = outputdirectory / output_graph_file_name

        output_chunks_file_name = f"chunks_{input_file_name[:-4]}.csv"
        output_chunks_file_with_path = outputdirectory / output_chunks_file_name

        output_context_prox_file_name = f"graph_contex_prox_{input_file_name[:-4]}.csv"
        output_context_prox_file_with_path = outputdirectory / output_context_prox_file_name

        # Generate documents from pages
        pages = data_generation.generate_docs_pages(input_file_name)
        df = data_generation.generate_docs2Dataframe(pages)
        print(df.shape)
        print(df.head())

        # Toggle to True if the time-consuming (re-)generation of the knowledge extraction is required
        if regenerate_data:
            results = df.apply(
                lambda row: self.execute_graph_prompt(row.text, self.client_model, {"chunk_id": row.chunk_id}), axis=1
            )
            concepts_list = data_processing.process_df2Graph(df, results)
            dfg1 = data_processing.process_graph2Df(concepts_list)

            if not os.path.exists(outputdirectory):
                os.makedirs(outputdirectory)

            dfg1.to_csv(output_graph_file_with_path, sep=";", index=False)
            df.to_csv(output_chunks_file_with_path, sep=";", index=False)
        else:
            dfg1 = pd.read_csv(output_graph_file_with_path, sep=";")

        # Clean up DataFrame
        dfg1.replace("", np.nan, inplace=True)
        dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
        dfg1['count'] = 4
        print(dfg1.shape)
        print(dfg1.head())

        # Process contextual proximity
        dfg2 = data_processing.proces_contextual_proximity(dfg1)
        dfg2.to_csv(output_context_prox_file_with_path, sep=";", index=False)
        print(dfg2.tail())

        # Combine dataframes
        dfg = pd.concat([dfg1, dfg2], axis=0)
        dfg = (
            dfg.groupby(["node_1", "node_2"])
            .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
            .reset_index()
        )
        print(dfg.head())

        # Create graph
        nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
        print(nodes.shape)

        G, communities = create_graph(nodes, dfg)
        colors = colors2Community(communities)

        for index, row in colors.iterrows():
            G.nodes[row['node']]['group'] = row['group']
            G.nodes[row['node']]['color'] = row['color']
            G.nodes[row['node']]['size'] = G.degree[row['node']]

        display_graph(G)

        return 'Successfully generated required Knowledge Graph.'


    def execute_Rag_Chatbot(self, query):
        """
        Execute Retrieval-Augmented Generation (RAG) chatbot task.
        """
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", data_generation.retrieve)  # retrieve
        workflow.add_node("grade_documents", rag_process.grade_documents)  # grade documents
        workflow.add_node("generate", rag_process.generate)  # generate
        workflow.add_node("transform_query", rag_process.transform_query)  # transform query
        workflow.add_node("web_search", rag_process.web_search)  # web search

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            rag_process.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        # Compile
        app = workflow.compile()

        inputs = {
            "keys": {
                "question": query,
            }
        }
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':")
            pprint.pprint("\n---\n")

        # Final generation
        pprint.pprint(value['keys']['generation'])