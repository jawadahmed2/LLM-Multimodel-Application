from data_preparation.prompt_template import Prompt_Template
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain import hub
from client.llm_connection import LLMConnection
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from typing_extensions import TypedDict, Dict
from typing import List
import os

class GraphState(TypedDict):
    keys: Dict[str, any]

class Grade(BaseModel):
    """Binary score for relevance check."""
    score: str = Field(description="Relevance score 'yes' or 'no'")

class RagProcess:
    def __init__(self):
        self.llm_connection = LLMConnection()
        self.prompt_template = Prompt_Template()
        self.chat_ollama = self.llm_connection.connect_chat_ollama()
        self.web_search_tool = TavilySearchResults(k=3, api_key=os.getenv("TAVILY_API_KEY"))

        self.parser = JsonOutputParser(pydantic_object=Grade)
        self.retrieval_grader = self._initialize_retrieval_grader()
        self.rag_chain = self._initialize_rag_chain()
        self.question_rewriter = self._initialize_question_rewriter()

    def _initialize_retrieval_grader(self):
        prompt = self.prompt_template.get_rag_prompt(self.parser)
        return prompt | self.chat_ollama | self.parser

    def _initialize_rag_chain(self):
        prompt = hub.pull("rlm/rag-prompt")
        return prompt | self.chat_ollama | StrOutputParser()

    def _initialize_question_rewriter(self):
        prompt = self.prompt_template.question_rewriter_prompt()
        return prompt | self.chat_ollama | StrOutputParser()

    def generate(self, state: GraphState) -> GraphState:
        print("---GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation
            }
        }

    def grade_documents(self, state: GraphState) -> GraphState:
        print("---CHECK RELEVANCE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        filtered_docs = []
        search = "No"
        for doc in documents:
            score = self.retrieval_grader.invoke({"question": question, "context": doc.page_content})
            if score["score"] == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                search = "Yes"

        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                "run_web_search": search,
            }
        }

    def transform_query(self, state: GraphState) -> GraphState:
        print("---TRANSFORM QUERY---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        better_question = self.question_rewriter.invoke({"question": question})
        print('Transform Query worked \n', better_question)
        return {
            "keys": {
                "documents": documents,
                "question": better_question,
            }
        }

    def web_search(self, state: GraphState) -> GraphState:
        print("---WEB SEARCH---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        try:
            docs = self.web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
            documents.append(web_results)
        except Exception as error:
            print(error)

        return {"documents": documents, "question": question}

    def decide_to_generate(self, state: GraphState) -> str:
        print("---DECIDE TO GENERATE---")
        state_dict = state["keys"]
        search = state_dict["run_web_search"]

        if search == "Yes":
            print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"
