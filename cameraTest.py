import cv2
import numpy as np
import subprocess # 외부 명령어(libcamera-vid)를 실행하여 카메라 데이터를 스트리밍
import shlex # 명령어 문자열을 안전하게 토큰화

# cmd 설명
# libcamera-vid: MJPEG 포맷으로 카메라 스트리밍
# --inline: 인코딩된 프레임을 연속적으로 출력
# --nopreview: 프리뷰 창을 띄우지 않음
# -t 0: 무한 스트리밍을 의미(0이 아닌 값이면 특정 시간 동안 실행)
# --codec mjpeg: MJPEG 포맷사용
# --framerate 30: 초당 30프레임
# -o -:표준 출력(stdout)으로 데이터를 보냄(-o <파일명>이 아니라 - 사용)
# --camera 0: 0번 카메라 장치 사용 
cmd = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 0'

# 외부 프로세스 실행
# subprocess.Popen()을 사용하면 위의 libcamera-vid 명령어를 실행
# stdout=subprocess.PIPE: 표준 출력을 파이프로 연결 -> 파이썬에서 데이터를 읽을 수 있도록 함
# stderr-subprocess.PIPE: 에러메시지도 받을 수 있도록 설정
process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#MJPEG 데이터 읽고 디코딩

try:
    buffer = b""
    while True:
        buffer += process.stdout.read(4096) # 4096바이트씩 읽음
        a = buffer.find(b'\xff\xd8') # JPEG 시작 마커 (SOI, Start of Image)
        b = buffer.find(b'\xff\xd9') # JPEG 종료 마커 (EOI, End of Image)

        if a != -1 and b != -1:
            jpg = buffer[a:b+2] # 하나의 JPEG 이미지 추출
            buffer = buffer[b+2:] # 다음 프레임을 위해 버퍼 업데이트 

            # cv2.imdecode()를 사용하여 JPEG 이미지를 OpenCV에서 처리할 수 있는 BGR 이미지로 변환
            bgr_frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if bgr_frame is not None:
                # 변환된 이미지를 cv2.imshow()로 화면에 출력
                cv2.imshow('Camera Stream', bgr_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
finally:
    #프로그램 종료시 프로세스 정리
    process.terminate()

