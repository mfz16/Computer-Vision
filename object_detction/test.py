import mediapipe as mp
import cv2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.solutions.hands
#hands = mp_hands.Hands()

#print(hands)

model_path = r'D://google_models/gesture/gesture_recognizer.task'
# Add the path to the system path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print(model_path)
model_asset_path = model_path
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
#r=mp.tasks.vision.HandLandmarksConnections

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    #print('gesture recognition result: {}'.format(result))
    #multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in result.hand_landmarks]
    image = output_image.numpy_view()
    top_gesture = result.gestures
    hand_landmarks = result.hand_landmarks
    print(top_gesture) 
    annotated_image = image.copy()
    print(cv2.imshow('original_frame', annotated_image))
    #image_h, image_w, _ = image.shape
   
    
    # Draw the hand landmarks on the image.
    for hand_landmark in hand_landmarks:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmark
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return annotated_image
    

# options = GestureRecognizerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     num_hands=2, min_hand_detection_confidence=0.4, result_callback=print_result)


base_options = python.BaseOptions(model_asset_path=model_path)
running_mode = vision.RunningMode.LIVE_STREAM
options = vision.GestureRecognizerOptions(base_options=base_options,running_mode=running_mode,result_callback=print_result)
recognizer = vision.GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, fr = cap.read()
    if not ret:
        break
    frame=fr.copy()
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    # Recognize hand gestures asynchronously
    new_image=recognizer.recognize_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    
 
    # Display the original frame
    cv2.imshow('original_frame', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print('No gesture recognition result')