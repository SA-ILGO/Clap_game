#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp

get_ipython().system('pip install keyboard')
get_ipython().system('pip install tensorflow')


# In[20]:


import cv2

# mediapipe ?‚¬?š©?•˜ê¸?
# ?† ì°¾ê¸° ê´?? ¨ ê¸°ëŠ¥ ë¶ˆëŸ¬?˜¤ê¸?
mp_hands = mp.solutions.hands
# ?† ê·¸ë ¤ì£¼ëŠ” ê¸°ëŠ¥ ë¶ˆëŸ¬?˜¤ê¸?
mp_drawing = mp.solutions.drawing_utils
# ?† ì°¾ê¸° ê´?? ¨ ?„¸ë¶? ?„¤? •
hands = mp_hands.Hands(
    max_num_hands = 2, # ?ƒì§??•  ìµœë?? ?†?˜ ê°??ˆ˜
    min_detection_confidence = 0.5, # ?‘œ?‹œ?•  ?†?˜ ìµœì†Œ ? •?™•?„
    min_tracking_confidence = 0.5 # ?‘œ?‹œ?•  ê´?? ˆ?˜ ìµœì†Œ ? •?™•?„
)

video = cv2.VideoCapture(0)

while video.isOpened() :
    ret, img = video.read()
    img = cv2.flip(img,1)
    # ?ŒŒ?´?¬?´ ?¸?‹ ?˜ ?•˜?„ë¡? BGR ?†’ RGBë¡? ë³?ê²?
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ?† ?ƒì§??•˜ê¸?
    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if not ret :
        break

    # ì°¾ì?? ?† ?‘œ?‹œ?•˜ê¸?
    if result.multi_hand_landmarks is not None :
        print(result.multi_hand_landmarks)
        # ?´ë¯¸ì???— ?† ?‘œ?˜„?•˜ê¸?
        for res in result.multi_hand_landmarks :
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    k = cv2.waitKey(30)
    if k == 49 :
        break
    cv2.imshow('hand', img)

video.release()
cv2.destroyAllWindows()


# In[21]:


import cv2
import mediapipe as mp
import numpy as np
import time, os, sys

actions = ['egg clap', 'wrist clap', 'fist clap', 'edge clap']
seq_length = 20 # ?›¹ ?¬ê¸?
secs_for_action = 30 # ?°?´?„° ?ˆ˜ì§? ?‹œê°? (ì´?)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        if(idx == 4): break
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] # visibility?Š” ?†ê°??½?´ ë³´ì´?Š”ì§? ?•ˆ ë³´ì´?Š”ì§? ?™•?¸

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

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)

    time.sleep(10)  # ?ŒŒ?¼ ?¬ê¸°ê?? ì»¤ì§ˆ ê²½ìš° time.sleep ê¸°ê°„?„ ?” ?¬ê²? ?„¤? • (?˜„?¬?Š” 30s)
    cap.release()
cv2.destroyAllWindows()



# In[2]:


import numpy as np
import os

# TensorFlowê°? ?‹¤?–‰?  ?•Œ ?´ë¥? ì°¸ì¡°?•˜?—¬ GPU ?‚¬?š© ë°? ë©”ëª¨ë¦? ?• ?‹¹?„ ? œ?–´
# ë©”ëª¨ë¦? ìµœì ?™”ë¥? ?œ„?•´ ?‚¬?š©
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# In[3]:


actions = [
    'egg clap',
    'wrist clap',
    'fist clap',
    'edge clap'
]

data = np.concatenate([
    np.load('dataset/seq_egg clap_1714829767.npy'),
    np.load('dataset/seq_wrist clap_1714829767.npy'),
    np.load('dataset/seq_fist clap_1714829767.npy'),
    np.load('dataset/seq_edge clap_1714829767.npy')
], axis=0)

data.shape


# In[4]:


x_data = data[:, :, :-1]
labels = data[:, 0, -1]

print(x_data.shape)
print(labels.shape)


# In[5]:


from tensorflow.keras.utils import to_categorical

# ?›?•«?¸ì½”ë”© ?‹¤?–‰
y_data = to_categorical(labels, num_classes=len(actions))
y_data.shape


# In[8]:


from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

# 10%ë¥? test_set
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()


# In[10]:


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    callbacks=[
        ModelCheckpoint('models/model.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)


# In[12]:


import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()


# In[13]:


from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.models import load_model

model = load_model('models/model.keras')

y_pred = model.predict(x_val)

multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))


# In[14]:


import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['clap1', 'clap2', 'clap3']
seq_length = 30

model = load_model('models/model.keras')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)

cap = cv2.VideoCapture(0)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
action_seq = []

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

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            # y_pred?˜ index, confidence ê°?? ¸?˜´
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            # ?„¸ ë²? ?™?•ˆ ?™?¼?•œ ?•¡?…˜?„ ë°˜ë³µ?• ?‹œ ?•´?‹¹ ëª¨ì…˜?œ¼ë¡? ?¸?‹ ( 
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        time.sleep(1)
        cap.release()
cv2.destroyAllWindows()



# In[6]:


import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['egg clap', 'wrist clap', 'fist clap', 'edge clap']
seq_length = 20

model = load_model('models/model.keras')

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

            wrist_x = res.landmark[0].x * img.shape[1]  # ?†ëª©ì˜ x ì¢Œí‘œ
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
            
            if right_hand_position - left_hand_position < 200:
                action = "actioning.."
                if action_left == action_right:
                    if action_left[-2] == action_left[-1] == action_left:
                        acrion = action_left
            else:
                action = "..."
                         
            cv2.putText(img, f'{action.upper()}', 
                        org=(int(win_w/2 - len(action.upper())*6), int(win_h/10)),  # ?ƒ?‹¨ ì¤‘ì•™ ?œ„ì¹? ì¡°ì •
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        cap.release()
cv2.destroyAllWindows()



# In[ ]:




