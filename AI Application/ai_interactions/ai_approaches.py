from task_execution.task_execution import Task_Execution
from gradio_chatbot.chat_interface import Interview_Bot
from loguru import logger

class AI_Approaches:
    def __init__(self):
        # Initialize the Task Execution class instance
        self.task_execution = Task_Execution()

    def automate_browsing(self, search_query: str):
        """
        Automate the browsing process for a specific search query.

        Returns:
            response (str): The response from the automated browsing execution.
        """
        # search_query = "What is happening with Arvind Kejriwal today?"
        logger.info(f"Automate Browsing with query {search_query}...")
        response = self.task_execution.execute_automate_browsing(search_query)
        return response['output']

    def interview_bot(self):
        """
        Launch the interview bot interface using Gradio.

        Returns:
            iface: The launched Gradio interface.
        """
        bot = Interview_Bot()
        iface = bot.gradio_interface()
        return iface.launch(share=True)

    def crewai(self):
        """
        Execute the CrewAI task.

        Returns:
            response (str): The response from the CrewAI execution.
        """
        logger.info("Execute CrewAI Task...")
        response = self.task_execution.execute_crewai()
        return response

    def get_image_information(self, image_path: str):
        """
        Get information about a specific image.

        Returns:
            response (str): The response containing the image information.
        """
        # image_path = "data_preparation/data/images/image.jpg"
        logger.info(f"Get Image Information for image {image_path}...")
        response = self.task_execution.get_image_informations(image_path)
        return response

    def get_instructions_training_data(self):
        """
        Generate training data for instructions.

        Returns:
            response (str): The response from the training data generation.
        """
        logger.info("Generate Instructions Training Data...")
        response = self.task_execution.generate_instructions_training_data(is_gen_instruct=True)
        return response

    def generate_knowledge_graph(self):
        """
        Generate a knowledge graph.

        Returns:
            response (str): The response from the knowledge graph generation.
        """
        logger.info("Generate Knowledge Graph...")
        response = self.task_execution.execute_knowledge_graph(regenerate_data=True)
        return response

    def powerful_rag_chatbot(self, query: str):
        """
        Execute the powerful RAG (Retrieval-Augmented Generation) chatbot.

        Returns:
            response (str): The response from the RAG chatbot execution.
        """
        # query = 'Explain how the different types of agent memory work?'
        logger.info(f"Execute RAG Chatbot with query {query}...")
        response = self.task_execution.execute_Rag_Chatbot(query)
        return response
