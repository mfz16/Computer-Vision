import cv2
import mediapipe as mp
import pybullet as p
import time
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Landmark indices mapping
landmarks_mapping = {
    "Wrist": 0,
    "Thumb_Base": 1,
    "Thumb_Knuckle": 2,
    "Thumb_Middle": 3,
    "Thumb_Tip": 4,
    "Index_Knuckle": 5,
    "Index_Middle": 6,
    "Index_Tip": 7,
    "Middle_Knuckle": 9,
    "Middle_Middle": 10,
    "Middle_Tip": 11,
    "Ring_Knuckle": 13,
    "Ring_Middle": 14,
    "Ring_Tip": 15,
    "Pinky_Knuckle": 17,
    "Pinky_Middle": 18,
    "Pinky_Tip": 19,
    "Palm_Base": 0,
    "Palm_Center": 12
}

# Disconnect from the existing PyBullet GUI connection if it exists
try:
    p.disconnect()
except:
    pass

# Connect to the PyBullet physics server with a new GUI connection
#physicsClient = p.connect(p.GUI)
physicsClient = p.connect(p.GUI, options="--opengl2 --cuda")

# Set gravity in the simulation
p.setGravity(0, 0, -9.8)

# Load a simple hand model into PyBullet
hand_urdf_path = "d://urdf_files/hand/urdf/shadow_hand.urdf"  # Replace with the actual path
hand_id = p.loadURDF(hand_urdf_path, useFixedBase=True)
for i in range(p.getNumJoints(hand_id)):
    joint_info = p.getJointInfo(hand_id, i)
    print(joint_info)
# Open webcam
cap = cv2.VideoCapture(0)

# Finger joint indices in PyBullet (adjust these based on your URDF file)
robot_finger_joint_indices = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,21]  # Adjust based on your URDF file
human_finger_joint_indices = [5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
robot_thumb_joint_indices = [2,3,4,5,6]
human_thumb_joint_indices = [1,2,3,4]
wrist_joint_index = 0  
forearm_joint_index = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    def calculate_joint_angle(landmark):
        x, y ,z= landmark.x, landmark.y,landmark.z
        return np.arctan2(z,np.sqrt(x**2+y**2))


#define set_joint_angle_function
    def set_joint_angles(joint_name, angles):
        joint_index = p.getJointInfo(hand_id, p.getJointIndex(hand_id, joint_name))[0]
        p.setJointMotorControl2(hand_id, joint_index, p.POSITION_CONTROL, targetPosition=angles)
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Control forearm-wrist joint based on the detected coordinates
        wrist_joint_angle = np.interp(hand_landmarks.landmark[0].y, [0, 1], [-0.5236, 0.1745])
        p.setJointMotorControl2(hand_id, jointIndex=0, controlMode=p.POSITION_CONTROL, targetPosition=wrist_joint_angle)
        
        #finger_joint_angle1 = np.interp(hand_landmarks.landmark[5].y, [0, 1], [-0.5236, 0.1745])
        finger_joint_angle1 = calculate_joint_angle(hand_landmarks.landmark[5])
        p.setJointMotorControl2(hand_id, jointIndex=7, controlMode=p.POSITION_CONTROL, targetPosition=finger_joint_angle1)
        
        #finger_joint_angle2 = np.interp(hand_landmarks.landmark[6].y, [0, 1], [-0.5236, 0.1745])
        finger_joint_angle2 = calculate_joint_angle(hand_landmarks.landmark[6])
        p.setJointMotorControl2(hand_id, jointIndex=8, controlMode=p.POSITION_CONTROL, targetPosition=finger_joint_angle2)
        
        #finger_joint_angle3 = np.interp(hand_landmarks.landmark[7].y, [0, 1], [-0.5236, 0.1745])
        
        finger_joint_angle3 = calculate_joint_angle(hand_landmarks.landmark[7])
        p.setJointMotorControl2(hand_id, jointIndex=9, controlMode=p.POSITION_CONTROL, targetPosition=finger_joint_angle3)
        
        #finger_joint_angle4 = np.interp(hand_landmarks.landmark[8].y, [0, 1], [-0.5236, 0.1745])
        finger_joint_angle4 = calculate_joint_angle(hand_landmarks.landmark[8])
        p.setJointMotorControl2(hand_id, jointIndex=10, controlMode=p.POSITION_CONTROL, targetPosition=finger_joint_angle4)
        
        
        # Control finger joints based on the detected coordinates
        #for i, joint_index in zip(human_finger_joint_indices,robot_finger_joint_indices):
            #finger_joint_angle = np.interp(hand_landmarks.landmark[i].y, [0, 1], [-0.5236, 0.1745])
            #p.setJointMotorControl2(hand_id, jointIndex=joint_index, controlMode=p.POSITION_CONTROL, targetPosition=finger_joint_angle)
        # Control thumb joints based on the detected coordinates
        # for joint_name, landmark_index in landmarks_mapping.items():
        #     joint_angle = np.interp(hand_landmarks.landmark[landmark_index].y, [0, 1], [-1.047, 1.047])
        #     try:
        #         joint_info = p.getJointInfo(hand_id, 1)
        #         print(joint_info)
        #     except Exception as e:
        #         print("Error getting joint information:", e)
        #     joint_index = next(j[0] for j in joint_info if j[1].decode() == joint_name.encode())
        #     p.setJointMotorControl2(hand_id, jointIndex=joint_index, controlMode=p.POSITION_CONTROL, targetPosition=joint_angle)

    # Step the PyBullet simulation
    p.stepSimulation()

    # Render the simulation
    p.getCameraImage(800, 600)

    # Display the frame
    cv2.imshow('Hand Gesture Simulation', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Disconnect from the PyBullet physics server
p.disconnect()

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
