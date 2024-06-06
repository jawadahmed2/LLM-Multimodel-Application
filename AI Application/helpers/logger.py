from loguru import logger
import logging
import sys

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where logged message originated,
        # skipping frames to find the correct function name and line number
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logger():
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    for _name in [
        "uvicorn",
        "gunicorn",
        "uvicorn.error",
        "gunicorn.error",
        "fastapi",
    ]:
        logging.getLogger(_name).handlers = [InterceptHandler()]

    for _name in logging.root.manager.loggerDict.keys():
        logging.getLogger(_name).handlers = [InterceptHandler()]

    # Define a custom format for the logger
    custom_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[request_id]}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    log_file = "helper/logs/backend.log"
    logger.configure(
        extra={"request_id": "app"}, # default identifer for request id
        handlers=[
            {"sink": sys.stdout, "level": "INFO", "format": custom_format},
            {
                "sink": log_file,
                "level": "DEBUG",
                "format": custom_format,
                "rotation": "1024 MB",
                "retention": "30 days",
            }
        ]
    )
