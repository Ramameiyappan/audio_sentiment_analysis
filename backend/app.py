from flask import Flask, request, jsonify
import tempfile
from flask_cors import CORS
from models import load_models
from emotion_pipeline import run_pipeline
import os
import subprocess

app = Flask(__name__)
CORS(app)

models = load_models()

@app.route("/health", methods=["GET"])
def health():
    try:
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return jsonify({"status": "ready", "ffmpeg": res.stdout.split('\n')[0]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    audio_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            audio_path = tmp.name

        timeline = run_pipeline(audio_path, models)

        return jsonify({
            "timeline": timeline
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)