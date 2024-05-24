from task_execution.task_execution import Task_Execution


task_execution = Task_Execution()

class AI_Approaches:
    def __init__(self):
        pass

    def automate_browsing(self):
        search_query = "What is happening with Arvind Kejriwal today?"
        response = task_execution.execute_automate_browsing(search_query)
        return response

    def interview_bot(self):
        iface = task_execution.gradio_interface()
        return iface.launch(share=True)

    def crewai(self):
        response = task_execution.execute_crewai()
        return response

    def get_image_information(self):
        image_path = "data_preparation/data/images/image.jpg"
        response = task_execution.get_image_informations(image_path)
        return response

    def get_instructions_training_data(self):
        response = task_execution.generate_instructions_training_data(is_gen_instruct=True)
        return response