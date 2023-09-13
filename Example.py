from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLOv8n model
model = YOLO('best_traffic_small_yolo.pt')

test_iamge_url = 'https://sanjosespotlight.s3.us-east-2.amazonaws.com/wp-content/uploads/2023/05/19153716/IMG_9313-scaled.jpg'

# Run inference on 'bus.jpg' with arguments
results = model.predict(source = test_iamge_url)

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image