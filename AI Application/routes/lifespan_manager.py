from fastapi_lifespan_manager import LifespanManager, State
from ai_interactions.ai_approaches import AI_Approaches
from loguru import logger
from fastapi import FastAPI
from typing import AsyncIterator


manager = LifespanManager()

select_ai_approach = AI_Approaches()

ml_models = {}


@manager.add
async def models_lifespan(app: FastAPI) -> AsyncIterator[State]:
    logger.info("Starting up and loading ML models...")
    # Load the ML models
    ml_models["automate_browsing"] = select_ai_approach.automate_browsing
    ml_models["crewai"] = select_ai_approach.crewai
    ml_models["get_image_information"] =select_ai_approach.get_image_information
    ml_models["get_instructions_training_data"] = select_ai_approach.get_instructions_training_data
    ml_models["generate_knowledge_graph"] = select_ai_approach.generate_knowledge_graph
    ml_models["powerful_rag_chatbot"] = select_ai_approach.powerful_rag_chatbot
    ml_models["interview_bot"] = select_ai_approach.interview_bot
    yield
    logger.info("Shutting down and releasing ML models...")
    # Clean up the ML models and release the resources
    ml_models.clear()