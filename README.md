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
## YOLO 이미지 객체 인식 시스템(YOLO Image Object Detection System)
* #### 이미지를 저장하지 않을 경우
  * sudo python3 yolo_object_detection_image.py --input 이미지 경로
    * 예) sudo python3 yolo_object_detection_image.py --input ./test_image/test_image_1.jpg
* #### 이미지를 저장할 경우
  * sudo python3 yolo_object_detection_image.py --input 이미지 경로 --output 저장할 이미지 경로
    * 예) sudo python3 yolo_object_detection_image.py --input ./test_image/test_image_1.jpg --output ./result_image/result_image_1.jpg

<div>
  <p align="center">
    <img width="300" src="test_image/test_image_1.jpg"> 
    <img width="300" src="result_image/result_image_1.jpg">
  </p>
</div>

<div>
  <p align="center">
    <img width="300" src="test_image/test_image_2.jpg"> 
    <img width="300" src="result_image/result_image_2.jpg">
  </p>
</div>

<div>
  <p align="center">
    <img width="300" src="test_image/test_image_3.jpg"> 
    <img width="300" src="result_image/result_image_3.jpg">
  </p>
</div>

---
## 실시간 YOLO 객체 인식 시스템(Real-Time YOLO Object Detection System) - 웹캠 또는 동영상(webcam or video)
* #### 비디오를 저장하지 않을 경우
  * webcam : sudo python3 real_time_yolo_object_detection.py
    * 예) sudo python3 real_time_yolo_object_detection.py
  * video : sudo python3 real_time_yolo_object_detection.py --input 비디오 경로
    * 예) sudo python3 real_time_yolo_object_detection.py --input ./test_video/test_video_1.mp4
* #### 비디오를 저장할 경우
  * webcam : sudo python3 real_time_yolo_object_detection.py --output 저장할 비디오 경로
    * 예) sudo python3 real_time_yolo_object_detection.py --output ./result_video/result_video_1.avi
  * video : sudo python3 real_time_yolo_object_detection.py --input 비디오 경로 --output 저장할 비디오 경로
    * 예) sudo python3 real_time_yolo_object_detection.py --input ./test_video/test_video_1.mp4 --output ./result_video/result_video_1.avi

<div>
  <p align="center">
    <img width="300" src="result_video/result_video_1.gif">
  </p>
</div>

<div>
  <p align="center">
    <img width="300" src="result_video/result_video_2.gif">
  </p>
</div>

<div>
  <p align="center">
    <img width="300" src="result_video/result_video_3.gif">
  </p>
</div>

##### 결과 영상 : https://drive.google.com/drive/folders/1yoWrHi6Rm4n5gmtoD3Ys1pww6VDbV9cJ?usp=sharing
---
