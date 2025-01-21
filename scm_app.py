import streamlit as st
import cohere
import pandas as pd
import datetime

# Initialize Cohere API
COHERE_API_KEY = "vPjqsnaockV4uYWjNcv56thveWsekn4D3jJVzBLE"  # Replace with your API key
co = cohere.Client(COHERE_API_KEY)

# Load the knowledge base
@st.cache_data
def load_knowledge_base(file_path):
    return pd.read_csv(file_path)

knowledge_base = load_knowledge_base("supply_chain_management.csv")  # Replace with your CSV file

# Function to query the knowledge base
def query_knowledge_base(query, knowledge_base, top_n=3):
    matches = knowledge_base[knowledge_base["content"].str.contains(query, case=False, na=False)]
    return matches.head(top_n).to_dict(orient="records")

# Function to get a response from Cohere's LLM
def get_cohere_response(prompt):
    response = co.generate(
        model='command-r-plus-08-2024',
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    return response.generations[0].text.strip()

# Initialize session state for messages and history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'history' not in st.session_state:
    st.session_state.history = {}

if 'selected_history' not in st.session_state:
    st.session_state.selected_history = None

# "New Conversation" button
if st.sidebar.button("New Conversation"):
    st.session_state.messages = []
    st.session_state.selected_history = None

# History section
st.sidebar.subheader("History")
for key in sorted(st.session_state.history.keys(), reverse=True):  # Show recent history first
    if st.sidebar.button(f"View {key}"):
        st.session_state.selected_history = key

# "Current Conversation" button
if st.sidebar.button("Current Conversation"):
    st.session_state.selected_history = None



# Main interface
st.title("Supply Chain Management Chatbot")

# Function to handle user input and generate response
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        # Query the knowledge base
        kb_results = query_knowledge_base(user_input, knowledge_base)

        # Generate a Cohere response
        prompt = f"You are a supply chain management assistant. Answer the following query based on general knowledge: {user_input}"
        cohere_response = get_cohere_response(prompt)

        # Combine results
        if kb_results:
            kb_response = "\n\n".join([f"- {res['content']}" for res in kb_results])
            final_response = f"{cohere_response}\n\nAdditionally, here's what we found in our knowledge base:\n{kb_response}"
        else:
            final_response = cohere_response

        # Save messages in session state
        st.session_state.messages.append(("User", user_input))
        st.session_state.messages.append(("Bot", final_response))

        # Save conversation to history
        conversation_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history[conversation_id] = st.session_state.messages[:]

        # Clear user input field
        st.session_state.user_input = ""

# Display selected history or ongoing conversation
if st.session_state.selected_history:
    st.subheader(f"Viewing History from: {st.session_state.selected_history}")
    selected_messages = st.session_state.history[st.session_state.selected_history]
    for i in range(0, len(selected_messages), 2):
        st.write(f"**You:** {selected_messages[i][1]}")
        st.write(f"**Bot:** {selected_messages[i + 1][1]}")
else:
   
    col1, col2 = st.columns([9, 1])
    with col1:
        st.text_input("You: ", key="user_input", label_visibility="collapsed", on_change=handle_input)
    with col2:
        st.button("⬆️", key="arrow_button", on_click=handle_input)

    for i in range(0, len(st.session_state.messages), 2):
        st.write(f"**You:** {st.session_state.messages[i][1]}")
        st.write(f"**Bot:** {st.session_state.messages[i + 1][1]}")
