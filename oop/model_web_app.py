import flask 
from flask import render_template, redirect, url_for
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(r'saved_model').to(DEVICE)

tokenizer = GPT2Tokenizer.from_pretrained(r'saved_model')

def generate_joke(start_joke: str) -> str:  

    text = start_joke
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    model.eval()

    with torch.no_grad():
        out = model.generate(input_ids, 
                            do_sample=True,
                            num_beams=2,
                            temperature=1.5,
                            top_p=0.9,
                            max_length=70,
                            )

    generated_text = list(map(tokenizer.decode, out))[0]
    extra_list = []


    for line in range(0, len(generated_text)-3):
        if (generated_text[line] == '[') and (generated_text[line + 1] == "S") and (generated_text[line + 2] == 'J') and (generated_text[line + 3] == ']'):
            break
        extra_list.append(generated_text[line])

    final_joke = "".join(extra_list) + "\n"

    return final_joke[:-6]


app = flask.Flask(__name__, template_folder= 'templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])

def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        start_joke = str(flask.request.form['get_joke'])
        result_joke = generate_joke(start_joke)
        return render_template('main.html', result = result_joke)

if __name__ == '__main__':
    app.run(debug=True)    



