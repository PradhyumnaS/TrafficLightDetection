import cv2
from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('models/best_traffic_small_yolo.pt')

# open a video file or start a video stream
cap = cv2.VideoCapture(0)  # replace with 0 for webcam

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # flip the image
    # frame = cv2.flip(frame, -1) 

    # Run inference on the current frame
    results = model(frame)  # results list

    for r in results:
        frame = r.plot()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Press 'q' on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object and destroy all windows
cap.release()
cv2.destroyAllWindows()