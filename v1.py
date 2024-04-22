#code to first detect objects in a video
from imageai.Detection import ObjectDetection

# Replace with your desired output path 
output_path = "D:/sample/observations.txt"

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("https://github.com/OlafenwaMoses/ImageAI/releases/download/v2.1.5/yolov3.weights")
detector.loadModel()

def process_video(video_path):
  with open("D:/sample", "w") as f:
    # Get video capture object
    cap = cv2.VideoCapture("D:/sample/video_shapes")

    while True:
      ret, frame = cap.read()
      if not ret:
        break

      # Detect objects in the frame
      detections = detector.detectObjectsFromImage(frame=frame, output_type="bounding_boxes", input_type="array", extract_detected_objects=True)

      # Write observations to file
      for detection in detections:
        name = detection["name"]
        percentage_probability = detection["percentage_probability"]
        box_points = detection["box_points"]
        f.write(f"Detected object: {name} with confidence: {percentage_probability:.2f}%\n"
                f"Bounding box: {box_points}\n")

    # Release resources
    cap.release()

# Replace with the path to your video file
video_path = "D:/sample/video_shapes.mp4"
process_video(video_path)

print(f"Finished processing video. Observations written to {output_path}")
