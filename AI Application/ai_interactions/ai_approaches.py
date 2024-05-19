from task_execution.task_execution import Task_Execution


task_execution = Task_Execution()

class AI_Approaches:
    def __init__(self):
        pass

    def automate_browsing(self):
        search_query = "What is happening with Arvind Kejriwal today?"
        response = task_execution.execute_automate_browsing(search_query)
        return response