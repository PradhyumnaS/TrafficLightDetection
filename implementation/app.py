from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import io
import os
import base64
import tempfile

app = Flask(__name__)

model = YOLO('models/best_traffic_small_yolo.pt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    file.save(temp_file.name)

    results = model.predict(source=temp_file.name)

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

        img_io = io.BytesIO()
        im.save(img_io, 'JPEG')
        img_io.seek(0)

        os.remove(temp_file.name)

        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        return render_template('result.html', result_image=img_base64)

    return "Error with image processing", 500

if __name__ == '__main__':
    app.run(debug=True)
