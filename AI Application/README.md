# AI Application

This project is an AI Application that uses various packages to facilitate interactions with language models, data preparation, and providing different services. The application is structured into multiple modules to handle different aspects of AI interactions, client connections, data preparation, and more.

## Setup Instructions

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system. Additionally, you will need Poetry to manage the project's dependencies.

### Install Poetry

If you don't have Poetry installed, you can install it using the following command:

```sh
pip install poetry
```

### Activate Poetry Environment

Remember in order to execute below command must be in project directory and deactivate any other virtual environment first

```sh
poetry shell
```

### Install the libraries once in the environment

```sh
poetry install
```

### Run the Application

```sh
python main.py
```

### Test Enpoints

In order to test the endpoints follow below link

```sh
http://localhost:8001/docs
```

## Project Structure

AI Application
├── ai_interactions
│ ├── ai_approaches.py
├── client
│ ├── llm_connection.py
├── config
│ ├── ai_config.py
│ ├── app_config.py
│ ├── HW_usage.py
├── data_preparation
│ ├── data
│ │ ├── images
│ │ │ └── image.jpg
│ │ ├── KGraph_data
│ │ │ ├── data_output
│ │ │ │ ├── chunks_Saxony_Eastern_Expansion_EP_96.csv
│ │ │ │ ├── graph_contex_prox_Saxony_Eastern_Expansion_EP_96.csv
│ │ │ │ └── graph_Saxony_Eastern_Expansion_EP_96.csv
│ │ │ ├── knowledge_graph.html
│ │ │ └── Saxony_Eastern_Expansion_EP_96.txt
│ │ ├── llm_tuning
│ │ ├── instructions.txt
│ │ ├── sample_train.jsonl
│ │ ├── sample_valid.jsonl
│ │ ├── train.jsonl
│ │ ├── train_valid.jsonl
│ │ └── valid.jsonl
│ ├── data_generation.py
│ ├── data_processing.py
│ ├── prompt_template.py
├── gradio_chatbot
│ ├── chat_interface.py
├── helpers
│ ├── logger.py
├── infographics
│ ├── generate_report.py
├── main.py
├── opdf_index
│ ├── index.faiss
│ └── index.pkl
├── project_packages
│ ├── project_packages
│ ├── pyproject.toml
│ ├── README.md
├── README.md
├── routes
│ ├── ai_routes.py
│ ├── lifespan_manager.py
└── task_execution
├── crewai_agent.py
├── graph_network.py
├── rag_chats.py
└── task_execution.py
