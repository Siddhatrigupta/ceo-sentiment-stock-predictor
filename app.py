from flask import Flask, render_template, request
from transformers import pipeline
import joblib

# Initialize Flask
app = Flask(__name__)

print("🚀 Loading models... please wait...")

# ------------------ LOAD MODELS ------------------
# 1️⃣ Sentiment Model (your fine-tuned model)
sentiment_pipe = pipeline("sentiment-analysis", model="./sentiment_model", tokenizer="./sentiment_model")

# 2️⃣ Tone Model (zero-shot)
tone_pipe = pipeline("zero-shot-classification", model="./tone_model")

# 3️⃣ Emotion Model (Transformer-based instead of NRCLex)
emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

# 4️⃣ ML Model for Stock Prediction
stock_model = joblib.load("stock_predictor.pkl")

# 5️⃣ Possible Tone Labels
tone_labels = ["optimistic", "cautious", "defensive", "visionary", "confident", "neutral"]

# ------------------ FLASK ROUTE ------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = ""
    emotion_result = ""
    tone_result = ""
    stock_prediction = ""

    if request.method == 'POST':
        text = request.form['text']

        # 1️⃣ Sentiment Analysis (fix numeric labels)
        sentiment = sentiment_pipe(text[:512])[0]
        label = sentiment['label']
        score = sentiment['score']

        if label in ["LABEL_0", "LABEL_1", "LABEL_2"]:
            if label == "LABEL_0":
                label_text = "Negative"
            elif label == "LABEL_1":
                label_text = "Neutral"
            else:
                label_text = "Positive"
        else:
            label_text = label.capitalize()

        sentiment_result = f"{label_text} ({score:.2f})"

        # 2️⃣ Emotion Detection (Transformer-based)
        emotion_pred = emotion_pipe(text[:512])[0]
        emotion_result = emotion_pred['label'].capitalize()

        # 3️⃣ Tone Classification
        tone_pred = tone_pipe(text, candidate_labels=tone_labels)
        tone_result = tone_pred['labels'][0].capitalize()

        # 4️⃣ 📈 Stock Prediction (ML model + logic correction)
        predicted_change = float(stock_model.predict([[score]])[0])

        # Logical consistency: sentiment vs prediction direction
        if label_text.lower() == "negative" and predicted_change > 0:
            predicted_change = -abs(predicted_change)
        elif label_text.lower() == "positive" and predicted_change < 0:
            predicted_change = abs(predicted_change)

        # Format output as percentage
        if predicted_change > 0:
            stock_prediction = f"📈 Predicted stock increase of approximately {predicted_change:.2f}%"
        elif predicted_change < 0:
            stock_prediction = f"📉 Predicted stock decrease of approximately {abs(predicted_change):.2f}%"
        else:
            stock_prediction = "⚖️ Stock price expected to remain stable"

    # Render results
    return render_template(
        'index.html',
        sentiment=sentiment_result,
        emotion=emotion_result,
        tone=tone_result,
        stock_prediction=stock_prediction
    )

# ------------------ RUN FLASK APP ------------------
if __name__ == '__main__':
    app.run(debug=True)
