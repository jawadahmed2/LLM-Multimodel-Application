from ai_interactions.ai_approaches import AI_Approaches
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from loguru import logger
from typing import Optional
from pydantic import BaseModel, Field


ai_router = APIRouter()

select_ai_approach = AI_Approaches()


class AutomateBrowsing(BaseModel):
    prompt: str

@ai_router.post("/ai/automate_browsing", tags=["AI Route"])
async def ai_automate_browsing(automate_browsing: AutomateBrowsing):
    """
    Endpoint to automate the browsing process for a specific search query.

    Expects a JSON payload with a 'prompt' field. The AI automates the browsing process
    based on the provided search query and returns the response as a JSON response.

    Returns:
    - JSON response containing the automated browsing response.
    - If 'prompt' is missing, returns an error response with status code 400.
    - If any exception occurs during browsing automation, returns an error response with status code 500.
    """
    try:
        search_query = automate_browsing.prompt
        response = select_ai_approach.automate_browsing(search_query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in browsing automation: {e}")
        raise HTTPException(status_code=500, detail="Error in browsing automation")


@ai_router.get("/ai/interview_bot", tags=["AI Route"])
async def ai_interview_bot():
    """
    Endpoint to launch the interview bot interface using Gradio.

    Returns:
    - The launched Gradio interface for the interview bot.
    """
    iface = select_ai_approach.interview_bot()
    return iface

class CrewAI(BaseModel):
    pass

@ai_router.post("/ai/crewai", tags=["AI Route"])
async def ai_crewai(crewai: CrewAI):
    """
    Endpoint to execute the CrewAI task.

    Returns:
    - JSON response containing the response from the CrewAI execution.
    - If any exception occurs during CrewAI execution, returns an error response with status code 500.
    """
    try:
        response = select_ai_approach.crewai()
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in CrewAI execution: {e}")
        raise HTTPException(status_code=500, detail="Error in CrewAI execution")


class ImageInformation(BaseModel):
    image_path: str

@ai_router.post("/ai/get_image_information", tags=["AI Route"])
async def ai_get_image_information(file: UploadFile = File(...)):
    """
    Endpoint to get information about a specific image.

    Expects an image file upload. The AI retrieves information about
    the image and returns the response as a JSON response.

    Returns:
    - JSON response containing the image information.
    - If any exception occurs during image information retrieval, returns an error response with status code 500.
    """
    try:
        # Save the uploaded file to a temporary location
        file_location = f"data_preparation/data/images/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Pass the file path to the AI approach
        response = select_ai_approach.get_image_information(file_location)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in image information retrieval: {e}")
        raise HTTPException(status_code=500, detail="Error in image information retrieval")

class InstructionsTrainingData(BaseModel):
    pass

@ai_router.post("/ai/generate_instructions_training_data", tags=["AI Route"])
async def ai_generate_instructions_training_data(instructions_training_data: InstructionsTrainingData):
    """
    Endpoint to generate training data for instructions.

    Returns:
    - JSON response containing the response from the training data generation.
    - If any exception occurs during training data generation, returns an error response with status code 500.
    """
    try:
        response = select_ai_approach.get_instructions_training_data()
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in training data generation: {e}")
        raise HTTPException(status_code=500, detail="Error in training data generation")

class KnowledgeGraph(BaseModel):
    pass

@ai_router.post("/ai/generate_knowledge_graph", tags=["AI Route"])
async def ai_generate_knowledge_graph(knowledge_graph: KnowledgeGraph):
    """
    Endpoint to generate a knowledge graph.

    Returns:
    - JSON response containing the response from the knowledge graph generation.
    - If any exception occurs during knowledge graph generation, returns an error response with status code 500.
    """
    try:
        response = select_ai_approach.generate_knowledge_graph()
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in knowledge graph generation: {e}")
        raise HTTPException(status_code=500, detail="Error in knowledge graph generation")


class PowerfulRAGChatbot(BaseModel):
    query: str

@ai_router.post("/ai/powerful_rag_chatbot", tags=["AI Route"])
async def ai_powerful_rag_chatbot(powerful_rag_chatbot: PowerfulRAGChatbot):
    """
    Endpoint to execute the powerful RAG (Retrieval-Augmented Generation) chatbot.

    Expects a JSON payload with a 'query' field. The AI executes the powerful RAG chatbot
    based on the provided query and returns the response as a JSON response.

    Returns:
    - JSON response containing the response from the powerful RAG chatbot execution.
    - If 'query' is missing, returns an error response with status code 400.
    - If any exception occurs during powerful RAG chatbot execution, returns an error response with status code 500.
    """
    try:
        query = powerful_rag_chatbot.query
        response = select_ai_approach.powerful_rag_chatbot(query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in powerful RAG chatbot execution: {e}")
        raise HTTPException(status_code=500, detail="Error in powerful RAG chatbot execution")