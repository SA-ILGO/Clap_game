from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model


app = Flask(__name__)
file_path1 = "Clap_game\static\js\clap_data_now.txt"
file_path2 = "Clap_game\static\js\clap_practice.txt"
file_path3 = "Clap_game\static\js\clap_rythem.txt"

def cognition_txt(clap):
    with open(file_path1, "a") as file:
        file.write(clap + '\n')

def cognition_now_txt(clap):
    with open(file_path2, "a") as file:
        file.write(clap + '\n')

def clap_rhythm(action):
    with open(file_path3, "a") as file:
        file.write(action + '\n')

def init_txt():
    with open(file_path1, "w") as file:
        file.write('')
    with open(file_path2, "w") as file:
        file.write('')
    with open(file_path3, "w") as file:
        file.write('')
        

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    init_txt()
    return 'Reset successful'
    
cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

def GenerateFrames():
    init_txt()

    actions = ['egg clap', 'wrist clap', 'fist clap', 'edge clap']
    seq_length = 20

    # model = load_model("Clap_game\models\model.keras")
    model = load_model("Clap_game\models\model2.keras")

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3)

    win_w, win_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    left_hand_seq = []  
    right_hand_seq = []
    action_seq = []

    flag = 0
    action = ""

    right_hand_position = 400
    left_hand_position = 0

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
                    v[[0,18,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                wrist_x = res.landmark[0].x * img.shape[1]  
                if wrist_x < img.shape[1]:
                    left_hand_position = wrist_x
                else:
                    right_hand_position = wrist_x

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
                                cognition_now_txt(action)
                                flag = 1
                elif flag == 1:
                    if right_hand_position - left_hand_position > 200:
                        flag = 0
                        cognition_txt(action)
                        action = "Clap!"
                            
                cv2.putText(img, f'{action.upper()}', 
                            org=(int(win_w/2 - len(action.upper())*6), int(win_h/10)),  
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        ret, buffer = cv2.imencode('.jpg', img)  
        frame = buffer.tobytes()  
        yield (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cv2.destroyAllWindows()

def GenerateFrames2():

    init_txt()
    action = "Ready..."

    flag2 = 0
    left_hand_position = 0
    right_hand_position = 400

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    win_w, win_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                wrist_x = res.landmark[0].x * img.shape[1]  
                if wrist_x < img.shape[1] / 2:
                    left_hand_position = wrist_x
                else:
                    right_hand_position = wrist_x

                if flag2 == 1:
                    if right_hand_position - left_hand_position < 150:
                        action = "Clap!!"
                        clap_rhythm(action)
                        flag2 = 0 
   
                elif flag2 == 0:
                    if right_hand_position - left_hand_position > 150:
                        action = "Ready..."
                        clap_rhythm(action)
                        flag2 = 1
                
                # clap_rhythm(action)
                            
                cv2.putText(img, f'{action.upper()}', 
                            org=(int(win_w/2 - len(action.upper())*6), int(win_h/10)),  
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, ), thickness=2)

        ret, buffer = cv2.imencode('.jpg', img)  
        frame = buffer.tobytes()  
        yield (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cv2.destroyAllWindows()

@app.route('/clap_memory', methods=["GET", "POST"])
def clap_memory():
      return render_template('clap_memory.html')  

@app.route('/rhythm_practice', methods=["GET", "POST"])
def rhythm_practice():
      return render_template('rhythm_practice.html')  

@app.route('/clap_rythem', methods=["GET", "POST"])
def clap_rythem():
      return render_template('clap_rythem.html') 

@app.route('/clap_practice', methods=["GET", "POST"])
def clap_practice():
      return render_template('clap_practice.html') 

@app.route('/memory_practice', methods=["GET", "POST"])
def memory_practice():
      return render_template('memory_practice.html') 

@app.route('/_', methods=["GET", "POST"])
def _():
      return render_template('_.html') 

@app.route('/stream')
def Stream():
      return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream2')
def Stream2():
      return Response(GenerateFrames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def main():
    return render_template('main.html')


if __name__ == "__main__":
      app.run(debug=True) 
