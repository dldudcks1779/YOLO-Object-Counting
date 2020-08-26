<div>
  <p align="center">
    <img width="800" src="result_video.gif"> 
  </p>
</div>

## YOLOv3(You Only Look Once v3)
* #### grid cell로 나누어 한 번에 클래스를 판단하고 통합하여 최종 객체를 판단
* #### Bounding Box Coordinate(좌표) 및 클래스 Classification(분류)을 동일 신경망 구조를 통해 동시에 실행
* #### 사람, 자전거, 자동차, 개, 고양이 등 약 80개의 레이블로 구성
  * ##### yolov3.weights 파일 : 사전 훈련된 네트워크 가중치
    * ##### 다운로드 : https://drive.google.com/drive/folders/1QnZHzsss3Jdz2QhvF3CKu0avBF7eMhlV?usp=sharing
  * ##### yolov3.cfg 파일 : 네트워크 구성
  * ##### coco.names 파일 : coco dataset에 사용된 80가지 클래스 이름
---
### 실행 환경
* #### Ubuntu
* #### OpenCV Version : 3.x.x
  * ##### 설치 : https://blog.naver.com/dldudcks1779/222020005648
* #### imutils
  * ##### 설치 : sudo pip3 install imutils
---
## YOLO 객체 카운팅 시스템(YOLO Object Counting System)
* #### 비디오를 저장하지 않을 경우
  * webcam : sudo python3 yolo_object_counting.py
    * 예) sudo python3 yolo_object_counting.py
  * video : sudo python3 yolo_object_counting.py --input 비디오 경로
    * 예) sudo python3 yolo_object_counting.py --input test_video.mp4
* #### 비디오를 저장할 경우
  * webcam : sudo python3 yolo_object_counting.py --output 저장할 비디오 경로
    * 예) sudo python3 yolo_object_counting.py --output result_video.avi
  * video : sudo python3yolo_object_counting.py --input 비디오 경로 --output 저장할 비디오 경로
    * 예) sudo python3 yolo_object_counting.py --input test_video.mp4 --output result_video.avi
---
