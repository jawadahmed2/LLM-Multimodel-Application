from ai_interactions.ai_approaches import AI_Approaches


select_ai_approach = AI_Approaches()

# Execute the automate browsing approach
# response = select_ai_approach.automate_browsing()
# print(response)


# Launch the interview bot
# iface = select_ai_approach.interview_bot()

# Launch the crewai
# response = select_ai_approach.crewai()
# print(f"""
#     Task completed!
#     Task: "Write python code for each user story to implement the image-to-video console app."
#     Output: {response}""")


# Get image information
# response = select_ai_approach.get_image_information()
# print(response)

# Generate instructions training data
# response = select_ai_approach.get_instructions_training_data()
# print(response)

# Generate knowledge graph
response = select_ai_approach.generate_knowledge_graph()
print(response)
