import io
import os

from flask import Flask, request, jsonify
import onnxruntime as ort

from utils import preprocess_image, postprocess

app = Flask(__name__)

# Load ONNX model once at startup
MODEL_PATH = os.path.join("best.onnx")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Traffic sign detection API"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        img0, img_tensor, r, dw, dh = preprocess_image(image_bytes)

        # Run inference
        outputs = session.run([output_name], {input_name: img_tensor})[0]

        detections = postprocess(outputs, img0.shape, r, dw, dh)

        labels = sorted(set(d["class_name"] for d in detections))

        return jsonify({
            "labels": labels
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=8080, debug=True)
