from .data_generation import Data_Generation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

data_generation = Data_Generation()

class Data_Processing:
    def __init__(self):
        pass

    def interview_bot_splitter(self, query):
        result = data_generation.generate_result(query)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_text(result)

        return doc_splits

    def store_docs_in_db(self, query):
        doc_splits = self.interview_bot_splitter(query)
        ollama_emb = OllamaEmbeddings(
                model="nomic-embed-text"
            )
        db = Chroma.from_texts(
            doc_splits,
            collection_name="rag-chroma",
            embedding=ollama_emb,
        )
        return db
