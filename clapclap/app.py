from flask import Flask, render_template, Response  # Flask 웹 애플리케이션과 템플릿 렌더링을 위한 모듈을 가져옵니다.
import cv2  # OpenCV 라이브러리를 가져옵니다.
import mediapipe as mp  # MediaPipe 라이브러리를 가져옵니다.
import numpy as np  # 배열 및 행렬 작업을 위한 NumPy 라이브러리를 가져옵니다.
from keras.models import load_model  # Keras 모델을 로드하기 위한 모듈을 가져옵니다.


app = Flask(__name__) #Flask 애플리케이션 객체 생성

file_path = "CLAPCLAP\static\js\clap_data.txt" #파일 경로 설정 

def cognition_txt(clap):
    with open(file_path, "a") as file:
        file.write(clap + '\n') #파일에 손뼉 정보 기록

def init_txt():
    with open(file_path, "w") as file:
        file.write('') #파일을 초기화

cap = cv2.VideoCapture(0)  # 웹캠으로부터 비디오 캡처 객체 생성
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 캡처된 비디오의 폭 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 캡처된 비디오의 높이 설정

def GenerateFrames():
    init_txt() #텍스트 파일을 초기화

    actions = ['egg clap', 'wrist clap', 'fist clap', 'edge clap'] #손뼉 정보 정의
    seq_length = 20 #입력 데이터의 시퀀스 길이 설정

    model = load_model("Clap\models\model.keras") #케라스 모델 로드

    # MediaPipe hands model
    mp_hands = mp.solutions.hands #mediapipe hands 모델 가져옴
    mp_drawing = mp.solutions.drawing_utils #mediapipe 라이브러리의 그리기 유틸리티 모듈 가져옴
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3)

    win_w, win_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT) #캡쳐된 비디오의 너비와 높이 가져옴
    
    left_hand_seq = []  # 왼손 시퀀스를 저장할 리스트를 초기화합니다.
    right_hand_seq = []  # 오른손 시퀀스를 저장할 리스트를 초기화합니다.

    flag = 0
    action = ""

    while cap.isOpened():
        ret, img = cap.read() #비디오 프레임 읽어오기
        img0 = img.copy()

        img = cv2.flip(img, 1) #이미지 좌우 반전
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #이미지 색상 공간 변경
        result = hands.process(img) #손 위치 감지
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #이미지 색상 공간을 다시 변경

        if result.multi_hand_landmarks is not None: #미디어파이프에서 손이 감지되었는지 확인
            for res in result.multi_hand_landmarks: #감지된 손에 대해 반복
                joint = np.zeros((21, 4)) #각 손의 21개 관절에 대한 배열을 초기화
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility] #손 관절을 joint 배열을 저장


                # 각 손의 관절 간의 벡터를 계산하여 손 각도 계산
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

                #손의 위치를 기준으로 왼손과 오른손을 분류
                wrist_x = res.landmark[0].x * img.shape[1]  
                if wrist_x < img.shape[1] / 2:
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
                

                #왼손 오른손의 동작이 일치하고, 일정 거리 이내에 위치한 경우에는 Clap! 동작을 인식하고 기록함
                if flag == 0:
                    if right_hand_position - left_hand_position < 200:
                        if action_left == action_right:
                            action = action_left
                            flag = 1
                elif flag == 1:
                    if right_hand_position - left_hand_position > 200:
                        flag = 0
                        cognition_txt(action)
                        action = "Clap!"
                            
                cv2.putText(img, f'{action.upper()}', 
                            org=(int(win_w/2 - len(action.upper())*6), int(win_h/10)),  
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        ret, buffer = cv2.imencode('.jpg', img)  # JPEG 형식으로 이미지를 인코딩합니다.
        frame = buffer.tobytes()  # 인코딩된 이미지를 바이트 스트림으로 변환합니다.
              # multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
        yield (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cv2.destroyAllWindows()



@app.route('/', methods=["GET", "POST"])
def Index():
      return render_template('index_segin.html')  # index.html 파일을 렌더링하여 반환합니다.

@app.route('/stream')
def Stream():
      # GenerateFrames 함수를 통해 비디오 프레임을 클라이언트에게 실시간으로 반환합니다.
      return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
      # 라즈베리파이의 IP 번호와 포트 번호를 지정하여 Flask 앱을 실행합니다.
      app.run()