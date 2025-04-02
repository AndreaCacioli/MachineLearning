from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def home():
    return "Ciaone"

@app.route('/bert')
@cross_origin()
def bert():
    import torch
    from transformers import AutoModel, AutoTokenizer

    bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    ## Input Japanese Text
    line = "吾輩は猫である。"

    inputs = tokenizer(line, return_tensors="pt")


    print(tokenizer.decode(inputs["input_ids"][0]))

    outputs = bertjapanese(**inputs)
    return outputs.__str__()


@app.route("/translation", methods = ["POST"])
@cross_origin()
def translate():
    print(request.form)
    if request.method == "POST":
        text = request.form["sentence"]
        print(text)
        return text
    else:
        return ""

if __name__ == "__main__":
    app.run(debug=True)
