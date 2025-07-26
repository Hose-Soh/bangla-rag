from flask import Flask, request, jsonify
from llm import agent

# Initialize the Flask app
app = Flask(__name__)

# Function to interact with the Gemini model
def interact(query: str) -> str:
    """
    Sends the query to the API and retrieves the generated response.
    """
    response = agent(query)
    return response

def process_response(response: str) -> str:
    """
    Processes and formats the response.
    """
    return response

def send_question(question: str) -> str:
    """
    Sends the question to the API and returns the processed response.
    """
    try:
        response = interact(question)
        return process_response(response)
    except Exception as e:
        return f"An error occurred: {e}"

# @app.route('/', methods=['GET'])
# def home():
#     return 'API is running. POST your query to /agent.'


@app.route('/agent', methods=['POST'])
def bangla_rag():
    """
    Flask route that handles incoming POST requests to generate responses
    using the agent model.
    """
    # Get the incoming question
    incoming_question = request.json.get('question', '').strip().lower()

    # Generate an answer using the model
    answer = send_question(incoming_question)

    # Return the answer as a JSON response
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
