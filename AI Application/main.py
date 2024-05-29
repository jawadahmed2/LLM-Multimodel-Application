from ai_interactions.ai_approaches import AI_Approaches
import config.HW_usage as HW_usage

HW_usage.set_hardware_usage()

def display_menu():
    print("""
    Please select a service to use:
    1. Automate Browsing
    2. Interview Bot
    3. CrewAI
    4. Get Image Information
    5. Generate Instructions Training Data
    6. Generate Knowledge Graph
    7. Powerful RAG Chatbot
    0. Exit
    """)

def main():
    select_ai_approach = AI_Approaches()

    while True:
        display_menu()
        choice = input("Enter your choice (0-7): ")

        if choice == '1':
            response = select_ai_approach.automate_browsing()
            print(response)
        elif choice == '2':
            iface = select_ai_approach.interview_bot()
        elif choice == '3':
            response = select_ai_approach.crewai()
            print(f"""
            Task completed!
            Task: "Write python code for each user story to implement the image-to-video console app."
            Output: {response}""")
        elif choice == '4':
            response = select_ai_approach.get_image_information()
            print(response)
        elif choice == '5':
            response = select_ai_approach.get_instructions_training_data()
            print(response)
        elif choice == '6':
            response = select_ai_approach.generate_knowledge_graph()
            print(response)
        elif choice == '7':
            response = select_ai_approach.powerful_rag_chatbot()
            print(response)
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
