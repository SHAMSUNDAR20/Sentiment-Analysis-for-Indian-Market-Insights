import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify

# Load pre-trained BERT model and tokenizer for sentiment analysis
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)

# Sentiment analysis function
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    sentiment = torch.argmax(logits).item()
    sentiments = ["very negative", "negative", "neutral", "positive", "very positive"]
    return sentiments[sentiment]

# API for sentiment analysis
@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.json
    input_text = data.get('text')
    sentiment_result = analyze_sentiment(input_text)
    return jsonify({'sentiment': sentiment_result})

if __name__ == "__main__":
    app.run(debug=True)
