import config.HW_usage as HW_usage
from fastapi import FastAPI
from config.app_config import get_app_config
from fastapi.middleware.cors import CORSMiddleware
from routes.ai_routes import ai_router
from helpers.logger import setup_logger
from routes.lifespan_manager import manager

app = FastAPI(lifespan=manager)

setup_logger()

HW_usage.set_hardware_usage()

config = get_app_config()
app.config = config
app.secret_key = "ai-application"

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register app routes
app.include_router(ai_router)


if __name__ == "__main__":
    import uvicorn

    # Run the application using Uvicorn server
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
