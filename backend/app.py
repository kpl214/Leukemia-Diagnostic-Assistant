from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_agent import run_agent
from tools import classify_image_tool
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        response = run_agent(prompt)
        return jsonify({"response": response})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    filename = secure_filename(image_file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(path)

    try:
        result = classify_image_tool.invoke({"image_path": path})
        return jsonify({"classification": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
