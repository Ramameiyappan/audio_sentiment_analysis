from flask import Flask, request, jsonify
import tempfile
from flask_cors import CORS
from models import load_models
from emotion_pipeline import run_pipeline
import os

app = Flask(__name__)
CORS(app)

models = load_models()

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        audio_path = tmp.name

    timeline = run_pipeline(audio_path, models)

    return jsonify({
        "timeline": timeline
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)