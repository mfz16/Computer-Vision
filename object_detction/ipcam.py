import cv2
import os
print(cv2.__version__)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
username = 'your username'
password = 'your password'
port='your port'
endpoint = ''       #endpoint = eg MAIN or Substream
ip = 'your ip'
#cap = cv2.VideoCapture(f'rtsp://{username}:{password}@{ip}/{endpoint}')
model = cv2.dnn.readNet('d://yolov3/yolov3.weights','d://yolov3/yolov3.cfg')
#model = cv2.dnn.readNetFromDarknet("d://yolov3/yolov3.cfg")


#cap = cv2.VideoCapture('rtspsrc location=<<rtsp://admin:@192.168.18.143:554/live/0/MAIN>> latency=0 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
#print(cap)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    print(cap.read())
    #Preprocess the input frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set the input for the YOLO model
    model.setInput(blob)
    
    # Get the output from the YOLO model
    output_layers = model.getUnconnectedOutLayersNames()
    outputs = model.forward(output_layers)
    
    # Process the output to get the detected objects
    
    ret, frame = cap.read()
    cv2.imshow('RTSP stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
cap.release()

cv2.destroyAllWindows()