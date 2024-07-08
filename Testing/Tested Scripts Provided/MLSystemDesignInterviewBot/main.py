import torch
from textwrap import fill
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import embeddings
import gradio as gr
from langchain import PromptTemplate
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import warnings
from IPython.display import display, Markdown

warnings.filterwarnings('ignore')
from langchain_community.llms import Ollama
import os


os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "mistral"


# Initialize the AI model with a specific configuration
ai_model = Ollama(model="mistral")


# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, torch_dtype=torch.float16,
#     trust_remote_code=True,
#     device_map="auto",
#     quantization_config=quantization_config
# )
# generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
# generation_config.max_new_tokens = 1024
# generation_config.temperature = 0.0001
# generation_config.top_p = 0.95
# generation_config.do_sample = True
# generation_config.repetition_penalty = 1.15

# pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     return_full_text=True,
#     generation_config=generation_config,
# )

# llm = HuggingFacePipeline(
#     pipeline=pipeline,
#     )
query = "Give me an indepth Recommendation System ML System Design"

llm = ai_model
result = llm(query)

display(Markdown(f"<b>{query}</b>"))
display(Markdown(f"<p>{result}</p>"))

# embeddings = HuggingFaceEmbeddings(
#     model_name="thenlper/gte-large",
#     model_kwargs={"device": "cuda"},
#     encode_kwargs={"normalize_embeddings": True},
# )


# db = Chroma.from_documents(texts_chunks, embeddings, persist_directory="db")


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_text(result)


db = Chroma.from_texts(
    doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'), persist_directory="db"
)

custom_template = """You are a Machine Learning System Design Interview help  AI Assistant. Given the
following conversation and a follow up question, Give an appropriate response with the ML context given to you/ '.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=db.as_retriever(search_kwargs={"k": 2}),
#     memory=memory,
#     condense_question_prompt=CUSTOM_QUESTION_PROMPT,
# )
# query = "Who you are?"
# result_ = qa_chain({"question": query})
# result = result_["answer"].strip()
# display(Markdown(f"<b>{query}</b>"))
# display(Markdown(f"<p>{result}</p>"))

def querying(query, history):
		memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
		qa_chain = ConversationalRetrievalChain.from_llm(
			llm=llm,
			retriever=db.as_retriever(search_kwargs={"k": 2}),
			memory=memory,
			condense_question_prompt=CUSTOM_QUESTION_PROMPT,
		)
		result = qa_chain({"question": query})
		return result["answer"].strip()


iface = gr.ChatInterface(
	fn = querying,
	chatbot=gr.Chatbot(height=600),
	textbox=gr.Textbox(placeholder="Tell me about Stripe System Design Articles?", container=False, scale=7),
	title="MLSystemDesignBot",
	theme="soft",
	examples=["How to design a System for Holiday Prediction like Doordash?",
						"Please summarize Expedia Group's Customer Lifetime Value Prediction Model"],
cache_examples=True,
		retry_btn="Retry",
		undo_btn="Undo",
		clear_btn="Clear",
		submit_btn="Submit"
		)
iface.launch(share=True)
