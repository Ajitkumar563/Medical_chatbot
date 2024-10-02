from flask import Flask, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import google.generativeai as palm
import os

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
palm.configure(api_key=GOOGLE_API_KEY)

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Define Pinecone index and search method
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})

# Define the system prompt
system_prompt = """
You are a helpful medical assistant. Provide clear and concise answers to medical questions.
"""

# Initialize the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Define a function to generate responses using PaLM API


def generate_palm_response(prompt_text):
    response = palm.generate_text(
        prompt=prompt_text,
        temperature=0.4,
        max_output_tokens=500
    )
    return response.result

# Create a wrapper class for the PaLM LLM to use with LangChain


class PaLM:
    def __init__(self, temperature=0.4, max_tokens=500):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, prompt):
        response = palm.generate_text(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
        return response.result


# Instantiate the PaLM LLM
llm = PaLM(temperature=0.4, max_tokens=500)

# Create chains for the retrieval-augmented generation (RAG) task
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Health check route


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# RAG API - Returns the response from the language model based on user input


@app.route("/rag", methods=["POST"])
def rag_query():
    try:
        # Get input from request body
        data = request.json
        if "query" not in data:
            return jsonify({"error": "Invalid request, 'query' field is missing."}), 400

        input_query = data["query"]

        # Process the query with RAG chain
        response = rag_chain.invoke({"input": input_query})

        # Return the generated answer
        return jsonify({"answer": response["answer"]}), 200

    except Exception as e:
        # Log and handle errors
        return jsonify({"error": str(e)}), 500


# Start the Flask application
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
