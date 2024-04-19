from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.json.get("text")
        vectorizer = joblib.load("./artifacts/vectorizer.sav")
        encoder = joblib.load("./artifacts/encoder.sav")
        model = joblib.load('./artifacts/model.sav')

        X = vectorizer.transform([text])
        pred = model.predict(X)
        res = encoder.inverse_transform(pred)[0]

        return jsonify({"message": f"{res}!"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)