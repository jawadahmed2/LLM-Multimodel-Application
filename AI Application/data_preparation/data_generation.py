from langchain import hub


class Data_Generation:
    def __init__(self):
        pass
    
    # Get the react prompting for the Automate Browsing Task
    @staticmethod
    def get_react_prompting():
        prompt = hub.pull("hwchase17/react")
        return prompt
