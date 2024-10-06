from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)

# Load the GPT-2 model and its tokenizer
generator = pipeline('text-generation', model='gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Set a pad token (using eos_token as pad_token)
tokenizer.pad_token = tokenizer.eos_token

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_text = data.get('input_text', '')
    max_length = data.get('max_length', 100)  # Ensure max_length is retrieved from the request

    # Generate text based on the input and max_length
    generated = generator(input_text, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    # Since the output is a list of dictionaries, access the first item and its 'generated_text' key
    generated_text = generated[0]['generated_text']
    
    return jsonify({'generated_text': generated_text})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
