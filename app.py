from flask import Flask, request, jsonify
import io
from PIL import Image
from src.model_se import SEModelTrainer
from src.model_eca import MySegmentationModel

app = Flask(__name)

# Define paths to your model files
fast_rcnn_model_path = 'trained_models/maskrcnn_eca.pth'
se_model_path = 'trained_models/maskrcnn_se.pth'

# Create instances of model trainers
fastrcnn_trainer = MySegmentationModel(fast_rcnn_model_path)
se_trainer = SEModelTrainer(se_model_path)

# Define routes for model predictions
@app.route('/predict/fastrcnn', methods=['POST'])
def predict_fastrcnn():
    try:
        image = request.files['image'].read()
        image = Image.open(io.BytesIO(image))

        predictions = fastrcnn_trainer.predict(image)

        # Process and return predictions as needed
        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict/se', methods=['POST'])
def predict_se():
    try:
        image = request.files['image'].read()
        image = Image.open(io.BytesIO(image))

        predictions = se_trainer.predict(image)

        # Process and return predictions as needed
        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
