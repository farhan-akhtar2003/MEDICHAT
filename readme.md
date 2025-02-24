
# Medical AI Chatbot
This is a medical AI chatbot that can help you with your medical queries. It is built using Python. The chatbot uses a deep learning model to predict the disease based on the symptoms provided by the user. It also provides information about the disease and possible treatments. The chatbot is designed to be user-friendly and easy to use. The chatbot is still in development and new features are being added regularly. If you have any feedback or suggestions, please feel free to contact me. We hope you find the chatbot helpful and informative. Thank you for using our medical AI chatbot!

## Project Explanation

### Retrieval-Augmented Generation (RAG) Model

The chatbot leverages a **RAG model** to deliver precise medical information:
- **Retrieval:** It fetches relevant information from a vector database built from the dataset "Current Medical Diagnosis and Treatment 2023 BY MAXINE A. PAPADAKIS, STEPHEN J. MCHPHEE, MICHAEL W. RABOW."
- **Generation:** A **Large Language Model (LLM)** generates responses based on the retrieved information.

### Large Language Model (LLM)

The **LLM** is trained on medical data, enabling it to understand user queries and produce accurate, human-like responses.

### Vector Database

A **Vector Database** stores numerical representations (embeddings) of the medical text. It performs similarity searches to retrieve the most relevant information for the user’s query.

### Workflow

1. **User Query:** The chatbot converts the user’s query into a vector.
2. **Information Retrieval:** The vector database retrieves the most similar documents.
3. **Response Generation:** The LLM generates a detailed response using the retrieved documents.
4. **User Interaction:** The chatbot provides the response to the user.

This architecture ensures that the chatbot delivers accurate, relevant, and context-aware medical information.

# Requirements
- Python 3.10 or higher
- anaconda installed

# Installation
1. Clone the repository
2. Install the required packages after creating a virtual environment
    ```bash
    conda deactivate
    clear
    conda create -p venv python=3.10
    conda activate venv/
    pip3 install -r requirements.txt
    ```
3. Run the chatbot
    ```bash
    streamlit run model.py
    ```
# Usage
1. Enter your symptoms
2. Get the predicted disease
3. Get information about the disease
4. Get possible treatments
5. Ask any other medical queries
6. Provide feedback or suggestions
7. Exit the chatbot
8. Thank you for using our medical AI chatbot!

# NOTE
Create a `.env` file and add the following environment variables:
```env
LANGCHAIN_API_KEY=YOUR_API_KEY
LANGCHAIN_PROJECT=YOUR_PROJECT_NAME
TOGETHER_API_KEY=YOUR_API_KEY

# FARHAN AKHTAR