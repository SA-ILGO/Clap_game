import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions = ['egg clap', 'wrist clap', 'fist clap', 'edge clap']
seq_length = 20

model = load_model("Clap_game\models\model.keras")

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)

cap = cv2.VideoCapture(0)
win_w, win_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

left_hand_seq = []  
right_hand_seq = []
action_seq = []

flag = 0
action = ""

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            wrist_x = res.landmark[0].x * img.shape[1]  
            if wrist_x < img.shape[1] / 2:
                left_hand_position = wrist_x
                hand_side = "Left"
            else:
                right_hand_position = wrist_x
                hand_side = "Right"

            d = np.concatenate([joint.flatten(), angle])

            if wrist_x < img.shape[1] / 2:  
                left_hand_seq.append(d)  
            else:
                right_hand_seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(left_hand_seq) < seq_length or len(right_hand_seq) < seq_length:
                continue

            input_data_left = np.expand_dims(np.array(left_hand_seq[-seq_length:], dtype=np.float32), axis=0)
            input_data_right = np.expand_dims(np.array(right_hand_seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred_left = model.predict(input_data_left).squeeze()
            y_pred_right = model.predict(input_data_right).squeeze()

            i_pred_left = int(np.argmax(y_pred_left))
            i_pred_right = int(np.argmax(y_pred_right))

            conf_left = y_pred_left[i_pred_left]
            conf_right = y_pred_right[i_pred_right]

            if conf_left < 0.8 or conf_right < 0.8:
                continue

            action_left = actions[i_pred_left]
            action_right = actions[i_pred_right]

            action_seq.append(action_left)

            if len(action_seq) < 5:
                continue
            
            if flag == 0:
                if right_hand_position - left_hand_position < 200:
                    action = "actioning..."
                    if action_left == action_right:
                        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                            action = action_left
                            flag = 1
            elif flag == 1:
                if right_hand_position - left_hand_position > 200:
                    action = "Clap!"
                    flag = 0

            
                         
            cv2.putText(img, f'{action.upper()}', 
                        org=(int(win_w/2 - len(action.upper())*6), int(win_h/10)),  
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        cap.release()
cv2.destroyAllWindows()

