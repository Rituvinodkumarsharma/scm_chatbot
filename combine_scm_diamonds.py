import streamlit as st
import pandas as pd
import cohere
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import time
from datetime import datetime

# Initialize Cohere API
COHERE_API_KEY = "vPjqsnaockV4uYWjNcv56thveWsekn4D3jJVzBLE"  # Replace with your API key
co = cohere.Client(COHERE_API_KEY)

# Load datasets
@st.cache_data
def load_knowledge_base(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_diamonds_data(file_path):
    return pd.read_csv(file_path)

knowledge_base = load_knowledge_base("supply_chain_management.csv")
diamonds_data = load_diamonds_data("diamonds.csv")

# Query the knowledge base
def query_knowledge_base(query, knowledge_base, top_n=3):
    matches = knowledge_base[knowledge_base["content"].str.contains(query, case=False, na=False)]
    return matches.head(top_n).to_dict(orient="records")

# Get Cohere response
def get_cohere_response(prompt):
    response = co.generate(
        model='command-r-plus-08-2024',
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    return response.generations[0].text.strip()

# Analyze diamonds data
def analyze_diamonds_data(query, data):
    columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']
    relevant_column = None

    # Match query to a relevant column
    for col in columns:
        if col.lower() in query.lower():
            relevant_column = col
            break

    if relevant_column:
        # Handle numeric columns
        if data[relevant_column].dtype in ['int64', 'float64']:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[relevant_column], bins=30, kde=True, ax=ax, color='blue')
            ax.set_title(f"Distribution of {relevant_column}")
            ax.set_xlabel(relevant_column)
            ax.set_ylabel("Frequency")

            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)

            # Generate description
            prompt = f"""
            You are a diamond industry assistant. Based on the following data trend:
            - Column: {relevant_column}
            - Summary Statistics:
                - Mean: {data[relevant_column].mean():.2f}
                - Median: {data[relevant_column].median():.2f}
                - Standard Deviation: {data[relevant_column].std():.2f}

            Write a short and meaningful insight about the column {relevant_column}.
            """

            response = co.generate(
                model='command-r-plus-08-2024',
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )

            description = response.generations[0].text.strip()

            return {
                "type": "graph",
                "description": description,
                "graph": img_buffer
            }

        # Handle categorical columns
        elif data[relevant_column].dtype == 'object':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=relevant_column, data=data, ax=ax, palette='viridis')
            ax.set_title(f"Distribution of {relevant_column}")
            ax.set_xlabel(relevant_column)
            ax.set_ylabel("Count")

            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)

            # Generate description
            top_value = data[relevant_column].value_counts().idxmax()
            top_count = data[relevant_column].value_counts().max()

            prompt = f"""
            You are a diamond industry assistant. Based on the following data trend:
            - Column: {relevant_column}
            - Most Common Value: {top_value} ({top_count} occurrences)

            Write a short and meaningful insight about the column {relevant_column}.
            """

            response = co.generate(
                model='command-r-plus-08-2024',
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )

            description = response.generations[0].text.strip()

            return {
                "type": "graph",
                "description": description,
                "graph": img_buffer
            }

    return {"type": "text", "description": "No relevant data found."}

# Streamlit app
st.title("Combined Chatbot App")
tabs = st.tabs(["Supply Chain Chatbot", "Diamond Insights Chatbot"])

# Tab 1: Supply Chain Chatbot
with tabs[0]:
    st.header("Supply Chain Management Chatbot")
    if 'supply_chain_messages' not in st.session_state:
        st.session_state.supply_chain_messages = []

    def handle_supply_chain_input():
        user_input = st.session_state.supply_chain_input
        if user_input:
            # Query the knowledge base
            kb_results = query_knowledge_base(user_input, knowledge_base)

            # Generate Cohere response
            prompt = f"You are a supply chain management assistant. Answer the following query based on general knowledge: {user_input}"
            cohere_response = get_cohere_response(prompt)

            # Combine results
            if kb_results:
                kb_response = "\n\n".join([f"- {res['content']}" for res in kb_results])
                final_response = f"{cohere_response}\n\nAdditionally, here's what we found in our knowledge base:\n{kb_response}"
            else:
                final_response = cohere_response

            st.session_state.supply_chain_messages.append(("You", user_input))
            st.session_state.supply_chain_messages.append(("Bot", final_response))
            st.session_state.supply_chain_input = ""

    st.text_input("You: ", key="supply_chain_input", on_change=handle_supply_chain_input)
    for sender, message in st.session_state.supply_chain_messages:
        st.write(f"**{sender}:** {message}")

# Tab 2: Diamond Insights Chatbot
with tabs[1]:
    st.header("Diamond Insights Chatbot")
    if 'diamond_messages' not in st.session_state:
        st.session_state.diamond_messages = []

    def handle_diamond_input():
        user_input = st.session_state.diamond_input
        if user_input:
            response = analyze_diamonds_data(user_input, diamonds_data)
            st.session_state.diamond_messages.append(("You", user_input))

            if response["type"] == "graph":
                st.session_state.diamond_messages.append(("Bot", response["description"], response["graph"]))
            else:
                st.session_state.diamond_messages.append(("Bot", response["description"], None))

            st.session_state.diamond_input = ""

    st.text_input("You: ", key="diamond_input", on_change=handle_diamond_input)
    for message in st.session_state.diamond_messages:
        if len(message) == 3 and message[2]:
            st.write(f"**{message[0]}:** {message[1]}")
            st.image(message[2])
        else:
            st.write(f"**{message[0]}:** {message[1]}")
