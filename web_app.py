from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Sentiment Analysis for Indian Market Insights"

if __name__ == "__main__":
    app.run(debug=True)