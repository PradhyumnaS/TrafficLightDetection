from tkinter import Tk, filedialog
from ultralytics import YOLO
from PIL import Image

model = YOLO('models/best_traffic_small_yolo.pt')

def upload_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    return file_path

image_path = upload_image()
if image_path:
    results = model.predict(source=image_path)

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
        im.save('results.jpg')
else:
    print("No image selected.")
