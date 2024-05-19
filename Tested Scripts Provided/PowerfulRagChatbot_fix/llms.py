from langchain_community.chat_models import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser


# Local Model
local_model = ChatOllama(model='mistral', temperature=0)

### Retrieval Grader
# Data model
class grade(BaseModel):
    """Binary score for relevance check."""

    score: str = Field(description="Relevance score 'yes' or 'no'")

parser = JsonOutputParser(pydantic_object=grade)

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved 
                    document to a user question. \n 
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, 
        grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out 
    erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the 
    document is relevant to the question. \n
    Provide the binary score as a JSON with no premable or 
    explaination and use these instructons to format the output: 
    {format_instructions}""",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

retrieval_grader = prompt | local_model | parser


### Rag Chain
# Prompt
prompt = hub.pull("rlm/rag-prompt")

# Chain
rag_chain = prompt | local_model | StrOutputParser()


### Question Re-writer
# Create a prompt template with format instructions and the query
prompt = PromptTemplate(
    template="""You are generating questions that is well optimized for 
                retrieval. \n 
    Look at the input and try to reason about the underlying sematic 
    intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Provide an improved question without any premable, only respond 
    with the updated question: """,
    input_variables=["question"],
)

# Prompt
question_rewriter = prompt | local_model | StrOutputParser()
