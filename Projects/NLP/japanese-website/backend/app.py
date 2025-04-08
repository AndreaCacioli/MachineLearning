from flask import Flask, render_template, request, Response
from flask_cors import CORS, cross_origin
import torch
from transformers import AutoModel, AutoTokenizer
import json

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def home():
    return "Ciaone"

@app.route("/translation", methods = ["POST"])
@cross_origin()
def translate():
    if request.method == "POST":
        japanese_sentence = request.form["sentence"]
        from transformers import MarianMTModel, MarianTokenizer
        # Load the tokenizer and model
        model_name = "Helsinki-NLP/opus-mt-ja-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Tokenize the input sentence
        inputs = tokenizer(japanese_sentence, return_tensors="pt", padding=True)

        # Perform the translation
        translated = model.generate(**inputs)

        # Decode the translated tokens to a string
        english_translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Example usage
        print(english_translation) 
        return Response(response=english_translation, status=200, content_type="text/plain")

    else:
        raise Exception("Translation error")

if __name__ == "__main__":
    app.run(debug=True)
