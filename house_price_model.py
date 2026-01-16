import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model
MODEL_PATH = "house_price_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded")

@app.route("/")
def home():
    return render_template("index.html", model_loaded=model is not None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", error="Model not loaded")

    try:
        rooms = float(request.form["rooms"])
        distance = float(request.form["distance"])

        if not (1 <= rooms <= 15):
            return render_template("index.html", error="Rooms must be 1–15")

        if not (0.1 <= distance <= 20):
            return render_template("index.html", error="Distance must be 0.1–20 km")

        X = pd.DataFrame([[rooms, distance]], columns=["Rooms", "Distance"])
        price = round(model.predict(X)[0], 2)

        return render_template(
            "index.html",
            prediction=f"${price} thousand",
            rooms=rooms,
            distance=distance,
            model_loaded=True
        )

    except Exception as e:
        return render_template("index.html", error=str(e))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    X = pd.DataFrame([[data["rooms"], data["distance"]]],
                     columns=["Rooms", "Distance"])
    prediction = model.predict(X)[0]

    return jsonify({"prediction": round(prediction, 2)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
