import cv2
import numpy as np
import subprocess
import shlex
import mediapipe as mp
import random
import time

# 초기 세팅
def init_settings():
    # 인식 가능한 11가지 동작
    gesture = {
        0: 'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
        6: 'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
    }

    # 가위바위보 동작 만들기
    rsp_gesture = {
        0: 'rock', 5:'paper', 9:'scissors', 1: 'scissors',
    }

    #MediaPipe 세팅
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # 손 찾기 관연 세부 설정
    hands = mp_hands.Hands(
        max_num_hands=1,  # 탐지할 최대 손의 객수
        min_detection_confidence=0.5,  # 표시할 손의 최소 정확도
        min_tracking_confidence=0.5  # 표시할 관절의 최소 정확도
    )

    # KNN 모델 로드
    file = np.genfromtxt('./data/gesture_train.csv', delimiter=',')
    angle = file[:,:-1].astype(np.float32)
    label = file[:,-1].astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    return gesture, rsp_gesture, mp_hands, mp_drawing, hands, knn

# libcamera-vid 프로세스 실행 및 프로세스 객체 반환
def start_camera_stream(camera_id = 0, width = 640, height=480, framerate= 10):
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width {width} --height {height} --framerate {framerate} -o - --camera {camera_id}'

    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

# libcamera에서 JPEG 데이터를 받아서 OpenCV 이미지로 디코딩 후 반환
def get_frame(process, buffer):
    buffer += process.stdout.read(4096) # 몇바이트씩 읽을껀지
    start_idx = buffer.find(b'\xff\xd8') # JPEG 시작 마커
    end_idx = buffer.find(b'\xff\xd9') # JPEG 종료 마커

    if start_idx != -1 and end_idx != -1:
        jpg = buffer[start_idx:end_idx+2] # JPEG 이미지 추출
        buffer = buffer[end_idx+2:] # 버퍼 업데이트

        img = cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8), cv2.IMREAD_COLOR)
        return img, buffer
    else:
        return None, buffer
    
# img에서 mediaPipeHands로 랜드마크 찾고, KNN 모델 통해서 가위바위보 동작 인식 후 반환
def detect_hand_gesture(img, hands, knn, rps_gesture,mp_drawing, mp_hands):
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # BGR -> RGB로 변환
    result = hands.process(rgb_img) # Mediapipe 손 탐지

    user = None

    # 손 인식 후 랜드마크 표시
    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x,lm.y,lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            # 유저가 무엇을 냈는지 반환
            idx = int(results[0][0])

            # 가위바위보 중 하나가 인식 되었을 경우
            if idx in rps_gesture:
                user = rps_gesture[idx]
                cv2.putText(
                    img, user.upper(),
                    (int(res.landmark[0].x * img.shape[1]), 
                     int(res.landmark[0].y * img.shape[0] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2
                )
            elif idx == 10:
                user = "OK"
                cv2.putText(
                    img,"OK",
                    (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2
                )


            mp_drawing.draw_landmarks(img,res, mp_hands.HAND_CONNECTIONS)

    return user

# 가위바위보 결과를 문자열로 반환
def decide_winner(user_rsp, computer_rsp):
    
    if not user_rsp:
        return None
    
    if user_rsp == computer_rsp:
        return "DRAW"
    elif (user_rsp == 'rock' and computer_rsp == 'scissors') or \
        (user_rsp == 'scissors' and computer_rsp == 'paper') or \
        (user_rsp == 'paper' and computer_rsp == 'rock'):
        return 'WIN'
    else:
        return 'LOSE'

# 상태에 따른 함수 실행
#게임 시작 전
def state_game_wait(img, current_gesture, state_data):
                
    if current_gesture == 'OK':
        state_data['state'] = 'start'
        state_data['computer_rsp'] = random.choice(['rock','paper','scissors'])
        print('com: ', state_data['computer_rsp'])

        # 음성 출력
        subprocess.Popen(shlex.split('mpg123 /home/srch/cogniciseToMediaPipe/mediaPipe/start.mp3'))
        state_data['game_start_time'] = time.time()
        state_data['image_saved'] = False
    return state_data

# 게임 시작했을 때 
def state_game_start(img,current_gesture, state_data):
    wait_time = time.time() - state_data['game_start_time']
    print(wait_time)
    if wait_time < 3.5:
        return state_data

    if not state_data['image_saved']:
        cv2.imwrite('user_rsp.png', img)
        state_data['image_saved'] = True
        print('이미지 저장 완료')
    
    # 사용자가 가위바위보중 하나를 냈을 경우
    if current_gesture in ['rock','paper','scissors']:
        state_data['user_rsp'] = current_gesture
        state_data['state'] = 'result'
        print('user: ', state_data['user_rsp'])

    return state_data

# 게임 끝났을때 
def state_game_result(img, current_gesture, state_data):

    result = decide_winner(state_data['user_rsp'], state_data['computer_rsp'])
    cv2.putText(img, result, (int(img.shape[1]/2) - 100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    cv2.putText(img,f"com: {state_data['computer_rsp']}",
                (int(img.shape[1]/2) - 100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),2)
    
    if current_gesture == 'OK':
        state_data['state'] = 'start'
        state_data['computer_rsp'] = random.choice(['rock','paper','scissors'])
        print('com: ', state_data['computer_rsp'])
        subprocess.Popen(shlex.split('mpg123 /home/srch/cogniciseToMediaPipe/mediaPipe/start.mp3'))
        state_data['game_start_time'] = time.time()
        state_data['image_saved'] = False
        state_data['user_rsp'] = None

    return state_data
        

def main():
    gesture, rsp_gesture, mp_hands, mp_drawing, hands, knn = init_settings()
    process = start_camera_stream(camera_id=0, width=640, height=480, framerate=10)
    buffer = b""

    state_data = {
        'state': 'wait',
        'computer_rsp': None,
        'user_rsp': None,
        'game_start_time': 0,
        'image_saved': False
    }

    try:
        while True:
            img, buffer = get_frame(process, buffer)
            if img is None:
                continue

            #좌우반전
            img = cv2.flip(img,1)

            # 손 제스처 인식
            current_gesture = detect_hand_gesture(img, hands, knn,rsp_gesture, mp_drawing, mp_hands)

           # 상태에 따라 실행
            if state_data['state'] == 'wait':
                state_data = state_game_wait(img,current_gesture, state_data)
            elif state_data['state'] == 'start':
                state_data = state_game_start(img,current_gesture, state_data)
            elif state_data['state'] == 'result':
                state_data = state_game_result(img,current_gesture, state_data)


            cv2.imshow('Game', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        process.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()