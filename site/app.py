from flask import Flask, request, render_template, Response
import random

import torch.nn as nn
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class Classifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.bert = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        # Also include followers and following
        self.layer2 = nn.Linear(2, 1, bias=True)
        
    def forward(self, x):
        out = torch.zeros((x.shape[0], 2))

        bert_out = self.bert(x[:, :512].long(), x[:, 512:1024].long()).to_tuple()[0]
        out[:, :2] = bert_out
        res = self.layer2(out)
        return torch.relu(res)
model = Classifier()

model.load_state_dict(torch.load("./model(2).bin"))


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        

def encode_sentence(sentence):
    res = torch.zeros((1, 1024))
    encoded = tokenizer(sentence, max_length=512, padding="max_length", truncation=True)
    res[0, :512] = torch.tensor(encoded['input_ids'])
    res[0, 512:1024] = torch.tensor(encoded['attention_mask'])
    return res

merged_file = "./data/merged_troll_data.json"

merged_df = pd.read_json(merged_file)

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(["png", "jpeg"])

app = Flask(__name__, template_folder="./templates", static_folder="./static")

tweets = []

troll_tweets = merged_df[merged_df["troll"] == True]['content'].values.tolist()
nontroll_tweets = merged_df[merged_df["troll"] == False]['content'].values.tolist()

@app.route("/")
def index():
    return render_template("index.html", tweets=tweets[::-1])

@app.route("/random")
def random_tweet():
    troll = request.args["type"] == "troll"
    
    tweet = random.choice(troll_tweets if troll else nontroll_tweets)
    return render_template("index.html", tweets=tweets[::-1], text=tweet)

@app.route("/submit", methods=["POST"])
def submit():
    global tweets
    tweet = request.form["tweet"]

    encoded = encode_sentence(tweet)
    output = model(encoded).item()
    print(output, flush=True)

    tweets.append((tweet, output >= 0.5))
    return render_template("index.html", tweets=tweets[::-1])