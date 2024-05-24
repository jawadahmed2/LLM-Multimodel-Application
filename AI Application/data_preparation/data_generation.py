from langchain import hub
from client.llm_connection import LLMConnection
import cv2
import base64
from io import BytesIO
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

llm_connection = LLMConnection()

class Data_Generation:
    def __init__(self):
        self.ollama = llm_connection.connect_ollama()

    # Get the react prompting for the Automate Browsing Task
    @staticmethod
    def get_react_prompting():
        prompt = hub.pull("hwchase17/react")
        return prompt

    def generate_result(self, query):
        result = self.ollama(query)
        return result

    # Function for reading image file from input directory and converting to frame crewai
    def read_image_and_convert_to_frame(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Open image using OpenCV
        return np.asarray(image)  # Convert the OpenCV matrix to NumPy array for processing

    # Function for generating the script content and writing to a file crewai
    def write_script_to_file(self, script_content: str, file_name: str) -> str:
        """
        Write the script content to a file.
        """
        # Extract script content between ```python and ```
        start_index = script_content.find("```python")
        l1 = len("```python")
        if start_index == -1:
            start_index = script_content.find("Action:")
            l1 = len("Action:")
        if start_index == -1:
            start_index = script_content.find("```bash")
            l1 = len("```bash")
        if start_index == -1:
            start_index = script_content.find("```markdown")
            l1 = len("```markdown")
        if start_index == -1:
            start_index = script_content.find("```makefile")
            l1 = len("```makefile")
        if start_index == -1:
            start_index = script_content.find("```text")
            l1 = len("```text")
        if start_index == -1:
            start_index = script_content.find("```")
            l1 = len("```")
        end_index = script_content.find("```", start_index + l1)
        if end_index == -1:
            end_index = len(script_content) - 1
        if start_index == -1 or end_index == -1:
            return "Script content must be enclosed between ```python and ``` tags."
        else:
            script_content = script_content[start_index + l1 : end_index].strip()

            # Write the extracted script content to the file
            file_path = Path(file_name)
            with open(file_path, "w") as file:
                file.write(script_content)
            return script_content


    def generate_base64_image(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")  # You can change the format if needed
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


    # load an online pdf and split it
    def load_doc(self) -> 'List[Document]':
        loader = OnlinePDFLoader("https://support.riverbed.com/bin/support/download?did=7q6behe7hotvnpqd9a03h1dji&version=9.15.0")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        return docs

    # vectorize, commit to disk and create a BM25 retriever
    def vectorize(self, embeddings) -> tuple[FAISS,BM25Retriever]:
        docs = self.load_doc()
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("./opdf_index")
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k=5
        return db,bm25_retriever

    # attempts to load vectorstore from disk
    def load_db(self) -> tuple[FAISS,BM25Retriever]:
        embeddings_model = HuggingFaceEmbeddings()
        try:
            db = FAISS.load_local("./opdf_index", embeddings_model)
            bm25_retriever = BM25Retriever.from_documents(load_doc())
            bm25_retriever.k=5
        except Exception as e:
            print(f'Exception: {e}\nno index on disk, creating new...')
            db,bm25_retriever = self.vectorize(embeddings_model)
        return db,bm25_retriever
