# LLM MultiModel Application

This project is an AI Application that uses various packages to facilitate interactions with language models, data preparation, and providing different services. The application is structured into multiple modules to handle different aspects of AI interactions, client connections, data preparation, and more.

## Setup Instructions

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system. Additionally, you will need Poetry to manage the project's dependencies.

### Install Poetry

If you don't have Poetry installed, you can install it using the following command:

```sh
pip install poetry==1.8.3
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

Note: Make sure to update the .env file in the config directory.

```sh
python main.py
```

### Test Enpoints

In order to test the endpoints follow below link

```sh
http://localhost:8001/docs
```


