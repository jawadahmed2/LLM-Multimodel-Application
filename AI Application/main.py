from ai_interactions.ai_approaches import AI_Approaches


select_ai_approach = AI_Approaches()

# Execute the automate browsing approach
# response = select_ai_approach.automate_browsing()
# print(response)


# Launch the interview bot
# iface = select_ai_approach.interview_bot()

# Launch the crewai
response = select_ai_approach.crewai()
print(f"""
    Task completed!
    Task: "Write python code for each user story to implement the image-to-video console app."
    Output: {response}""")
