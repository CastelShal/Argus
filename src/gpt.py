import cv2
import threading
from deepface import DeepFace
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to process frames from a single camera feed
def process_camera_feed(camera_id, embeddings_db=[], threshold=0.4):
    cap = cv2.VideoCapture(camera_id)

    with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        return

    print(f"Processing camera feed {camera_id}...")

    # Initialize the Dlib face detector
    # detector = dlib.get_frontal_face_detector()

    # Initialize the tracker (use CSRT for better accuracy and robustness)
    tracker = cv2.TrackerCSRT_create()  # You can also try others like KCF or MIL

    # Variable to track face region (bounding box)
    face_bbox = None
    tracking = False    

    while True:  # Check the flag to continue or stop
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_id}")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Dlib's face detector
        if not tracking:
            faces = DeepFace.extract_faces(frame, detector_backend='mediapipe', enforce_detection=False)
            
            if len(faces) > 0:
                # Use the first detected face
                face1 = faces[0]['facial_area']
                x, y, w, h, _, _ = face1.values()
                face_bbox = (x, y, w, h)
                tracker.init(frame, face_bbox)  # Initialize the tracker with the first face
                tracking = True
        else:
            # Update the tracker and get the new position of the face
            success, face_bbox = tracker.update(frame)
            if not success:
                tracking = False
                face_bbox = None

        # Perform face recognition if tracking
        if face_bbox is not None:
            x, y, w, h = face_bbox
            rgb_face = frame[y:y+h, x:x+w]  # Crop the detected face region

            # Perform facial recognition with DeepFace
            # match = recognize_faces(rgb_face, embeddings_db, threshold)

            # # Annotate the frame with recognition result
            # if match:
            #     name, distance = match
            #     cv2.putText(frame, f"{name} ({distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw the bounding box around the tracked face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the camera feed
        cv2.imshow(f"Camera {camera_id}", frame)

        # Break the loop with 'q' for all feeds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {camera_id}")
    print(f"Camera feed {camera_id} terminated.")

# Function to perform face recognition
# def recognize_faces(frame, embeddings_db, threshold=0.4):
#     try:
#         # Extract face embedding
#         embedding = DeepFace.represent(img_path=frame, model_name="Facenet")[0]["embedding"]

#         # Compare with database embeddings
#         results = []
#         for entry in embeddings_db:
#             name, db_embedding = entry['name'], np.array(entry['embedding'])
#             distance = np.linalg.norm(np.array(embedding) - db_embedding)
#             if distance < threshold:
#                 results.append((name, distance))

#         # Sort results by distance
#         results = sorted(results, key=lambda x: x[1])
#         return results[0] if results else None
#     except Exception as e:
#         print("Error in recognition:", e)
#         return None

# # Load a mock database of embeddings (replace with real data)
# # embeddings_db = [
# #     {"name": "Person1", "embedding": [/* Precomputed embedding for Person1 */]},
# #     {"name": "Person2", "embedding": [/* Precomputed embedding for Person2 */]},
# # ]

# # Camera feed IDs (modify as needed)
# camera_feeds = [0, 1, 2]  # Example: Three camera feeds

# # Start threads for each camera feed
# # threads = []
# # for camera_id in camera_feeds:
# #     # Start the camera feed thread
# #     thread = threading.Thread(target=process_camera_feed, args=(camera_id, embeddings_db))
# #     threads.append(thread)
# #     thread.start()

# # # Wait for all threads to finish
# # try:
# #     for thread in threads:
# #         thread.join()
# # except KeyboardInterrupt:
# #     print("Program interrupted. Closing all feeds...")

process_camera_feed("video.mp4")
# process_camera_feed("http://192.168.0.105:8080/video")
