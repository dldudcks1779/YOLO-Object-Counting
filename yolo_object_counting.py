##### 실행 #####
# 비디오를 저장하지 않을 경우
# webcam : sudo python3 yolo_object_counting.py
# 예) sudo python3 yolo_object_counting.py
# video : sudo python3 yolo_object_counting.py --input 비디오 경로
# 예) sudo python3 yolo_object_counting.py --input test_video.mp4
#
# 비디오를 저장할 경우
# webcam : sudo python3 yolo_object_counting.py --output 저장할 비디오 경로
# 예) sudo python3 yolo_object_counting.py --output result_video.avi
# video : sudo python3yolo_object_counting.py --input 비디오 경로 --output 저장할 비디오 경로
# 예) sudo python3 yolo_object_counting.py --input test_video.mp4 --output result_video.avi

# 필요한 패키지 import
from imutils.video import FPS
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import cv2 # opencv 모듈
import os # 운영체제 기능 모듈

# 실행을 할 때 인자값 추가
ap = argparse.ArgumentParser() # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
ap.add_argument("-o", "--output", type=str, help="output 비디오 경로") # 비디오 저장 경로
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="최소 확률")
# 퍼셉트론 : 입력 값과 활성화 함수를 사용해 출력 값을 다음으로 넘기는 가장 작은 신경망 단위
# 입력 신호가 뉴런에 보내질 때 가중치가 곱해짐
# 그 값들을 더한 값이 한계값을 넘어설 때 1을 출력
# 이 때 한계값을 임계값이라고 함
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="임계값")
# 입력받은 인자값을 args에 저장
args = vars(ap.parse_args())

# YOLO 모델이 학습된 coco 클래스 레이블
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# YOLO 가중치 및 모델 구성에 대한 경로
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"]) # 가중치
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"]) # 모델 구성

# COCO 데이터 세트(80 개 클래스)에서 훈련된 YOLO 객체 감지기 load
print("[YOLO 객체 감지기 loading...]")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# YOLO에서 필요한 output 레이어 이름
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# input 비디오 경로가 제공되지 않은 경우 webcam
if not args.get("input", False):
    print("[webcam 시작]")
    vs = cv2.VideoCapture(0)

# input 비디오 경로가 제공된 경우 video
else:
    print("[video 시작]")
    vs = cv2.VideoCapture(args["input"])

# fps 정보 초기화
fps = FPS().start()

writer = None
(W, H) = (None, None)

# 전체 프레임 수
totalFrame = 0

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()

    # 읽은 프레임이 없는 경우 종료
    if args["input"] is not None and frame is None:
        break
    
    # 프레임 크기 지정
    frame = imutils.resize(frame, width=1000)

    # 프레임 크기
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    # 전체 프레임 수 1 증가
    totalFrame += 1
    
    # 5개의 프레임 마다 객체 인식
    if totalFrame % 5 == 1:
        # blob 이미지 생성
        # 파라미터
        # 1) image : 사용할 이미지
        # 2) scalefactor : 이미지 크기 비율 지정
        # 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    
        # 객체 인식
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # bounding box, 확률 및 클래스 ID 목록 초기화
        boxes = []
        confidences = []
        classIDs = []
    
    # counting 수 초기
    count = 0
    
    # layerOutputs 반복
    for output in layerOutputs:
        # 각 클래스 레이블마다 인식된 객체 수 만큼 반복
        for detection in output:
            # 인식된 객체의 클래스 ID 및 확률 추출
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # 사람이 아닌 경우 제외
            if classID != 0: # # yolo-coco 디렉터리에 coco.names 파일을 참고하여 다른 object 도 인식 가능(0 인 경우 사람)
                continue
            
            # 객체 확률이 최소 확률보다 큰 경우
            if confidence > args["confidence"]:
                # bounding box 위치 계산
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int") # (중심 좌표 X, 중심 좌표 Y, 너비(가로), 높이(세로))
                
                # boun화ding box 왼쪽 위 좌표
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # bounding box, 확률 및 클래스 ID 목록 추가
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # bounding box가 겹치는 것을 방지(임계값 적용)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    
    # 인식된 객체가 있는 경우
    if len(idxs) > 0:
        # 모든 인식된 객체 수 만큼 반복
        for i in idxs.flatten():
            # counting 수 증가
            count += 1

            # bounding box 좌표 추출
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # bounding box 출력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 클래스 ID 및 확률
            text = "{} : {:.2f}%".format(LABELS[classIDs[i]], confidences[i])

            # label text 잘림 방지
            y = y - 15 if y - 15 > 15 else y + 15
            
            # text 출력
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # counting 결과 출력
    counting_text = "People Counting : {}".format(count)
    cv2.putText(frame, counting_text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # 프레임 출력
    cv2.imshow("Real-Time Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # 'q' 키를 입력하면 종료
    if key == ord("q"):
        break
    
    # fps 정보 업데이트
    fps.update()
    
    # output video 설정
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    
    # 비디오 저장
    if writer is not None:
        writer.write(frame)

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 종료
vs.release()
cv2.destroyAllWindows()
