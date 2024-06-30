from client.llm_connection import LLMConnection
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import gradio as gr
from langchain_community.vectorstores import Chroma
from data_preparation import prompt_template
from langchain_community import embeddings

llm_connection = LLMConnection()

class Interview_Bot:
    """
    A class representing an interview bot.

    Attributes:
    - ollama: The connection to the LLM model.
    """

    def __init__(self):
        """
        Initialize the Interview_Bot class.
        """
        self.ollama = llm_connection.connect_ollama()

    def gradio_interface(self):
        """
        Create a Gradio chat interface for the interview bot.

        Returns:
        - iface: Gradio chat interface.
        """
        iface = gr.ChatInterface(
            fn=self.querying,
            chatbot=gr.Chatbot(height=600),
            textbox=gr.Textbox(
                placeholder="Tell me about Stripe System Design Articles?",
                container=False,
                scale=7
            ),
            title="MLSystemDesignBot",
            theme="soft",
            examples=[
                "How to design a System for Holiday Prediction like Doordash?",
                "Please summarize Expedia Group's Customer Lifetime Value Prediction Model"
            ],
            cache_examples=True,
            retry_btn="Retry",
            undo_btn="Undo",
            clear_btn="Clear",
            submit_btn="Submit"
        )

        return iface

    def querying(self, query, history):
        """
        Query the LLM model for a response.

        Args:
        - query: The user's query.
        - history: The chat history.

        Returns:
        - result["answer"].strip(): The response from the LLM model.
        """
        db = self.store_docs_in_db("Give me an indepth Recommendation System ML System Design")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.ollama,
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            condense_question_prompt=prompt_template.interview_bot_template(),
        )
        result = qa_chain({"question": query})
        return result["answer"].strip()

    def store_docs_in_db(self, query):
        """
        Store documents in a database.

        Args:
        - query: The query for document splitting.

        Returns:
        - db: The database containing the documents.
        """
        doc_splits = self.interview_bot_splitter(query)
        ollama_emb = embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')
        db = Chroma.from_texts(
            doc_splits,
            collection_name="rag-chroma",
            embedding=ollama_emb, persist_directory="db"
        )
        return db
