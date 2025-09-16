#!/usr/bin/env python3
# people_count_current_frame.py
# 한 프레임에 보이는 사람 수만 실시간으로 표시합니다.
# 사용법 예:
#   python people_count_current_frame.py --source 0 --model yolo
#   python people_count_current_frame.py --source video.mp4 --model haar --no-display

import cv2
import argparse
import time
import numpy as np
import imutils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                        help="웹캠 인덱스(숫자) 또는 비디오 파일/RTSP URL (기본=0)")
    parser.add_argument("--model", choices=["yolo", "haar"], default="yolo",
                        help="검출 모델: yolo 또는 haar (기본=yolo). yolo 미설치 시 자동으로 haar로 폴백")
    parser.add_argument("--min-conf", type=float, default=0.35,
                        help="YOLO 최소 신뢰도 (confidence threshold)")
    parser.add_argument("--nms-thresh", type=float, default=0.4,
                        help="NMS IoU 임계값 (기본 0.4)")
    parser.add_argument("--no-display", action="store_true",
                        help="화면 출력 비활성화")
    return parser.parse_args()

# --- Haar Cascade detector ---
class HaarDetector:
    def __init__(self, cascade_path=None, minSize=(60,60)):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.minSize = minSize

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=self.minSize)
        boxes = []
        scores = []
        for (x, y, w, h) in rects:
            boxes.append([int(x), int(y), int(w), int(h)])  # x,y,w,h for NMSBoxes
            scores.append(1.0)  # Haar는 신뢰도 미제공 -> 동일 점수
        return boxes, scores

# --- YOLO (ultralytics) detector ---
class YOLODetector:
    def __init__(self, conf=0.35, device='cpu'):
        from ultralytics import YOLO
        # 경량 모델 사용 (자동으로 다운로드 가능)
        self.model = YOLO("yolov8n.pt")
        self.conf = conf

    def detect(self, frame):
        # ultralytics의 model()은 numpy array 입력 가능
        results = self.model(frame, imgsz=640, conf=self.conf)[0]
        boxes = []
        scores = []
        # results.boxes: xyxy, cls, conf
        if hasattr(results, "boxes") and len(results.boxes) > 0:
            b = results.boxes.xyxy.cpu().numpy()    # [[x1,y1,x2,y2], ...]
            cls = results.boxes.cls.cpu().numpy()   # class ids
            confs = results.boxes.conf.cpu().numpy()# confidences
            for box, c, conf in zip(b, cls, confs):
                if int(c) == 0:  # COCO person class id == 0
                    x1, y1, x2, y2 = box.astype(int)
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append([int(x1), int(y1), int(w), int(h)])  # convert to x,y,w,h
                    scores.append(float(conf))
        return boxes, scores

def apply_nms(boxes, scores, nms_thresh):
    # OpenCV expects boxes as [x,y,w,h] and scores list
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=nms_thresh)
    # cv2.dnn.NMSBoxes returns list of indices or array; normalize to flat list
    if isinstance(indices, (list, tuple)):
        idxs = [i for i in indices]
    else:
        idxs = indices.flatten().tolist() if indices.size else []
    final = [boxes[i] for i in idxs]
    final_scores = [scores[i] for i in idxs]
    return final, final_scores

def main():
    args = parse_args()
    src = args.source
    if src.isdigit():
        src = int(src)

    # 모델 초기화 시도 (yolo 우선)
    detector = None
    if args.model == "yolo":
        try:
            detector = YOLODetector(conf=args.min_conf)
            using = "YOLO"
        except Exception as e:
            print("YOLO 초기화 실패 (ultralytics 미설치 또는 오류). Haar Cascade로 폴백합니다.\n오류:", e)
            detector = HaarDetector()
            using = "Haar"
    else:
        detector = HaarDetector()
        using = "Haar"

    print(f"[INFO] Detector: {using}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] 영상 소스를 열 수 없습니다:", args.source)
        return

    time.sleep(0.3)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 첫 프레임을 읽을 수 없습니다.")
        cap.release()
        return

    W = frame.shape[1]
    H = frame.shape[0]
    fps_start = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=W)

        boxes, scores = detector.detect(frame)  # boxes: x,y,w,h

        # NMS로 중복 제거
        if len(boxes) > 0:
            final_boxes, final_scores = apply_nms(boxes, scores, args.nms_thresh)
        else:
            final_boxes, final_scores = [], []

        # 현재 프레임에 보이는 사람 수
        current_count = len(final_boxes)

        # 시각화: 바운딩박스와 카운트 표시
        for (b, s) in zip(final_boxes, final_scores):
            x, y, w, h = b
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"person {s:.2f}"
            cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.putText(frame, f"Current: {current_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        # FPS 계산 (단순 슬라이딩)
        frame_count += 1
        if frame_count >= 10:
            now = time.time()
            fps = frame_count / (now - fps_start)
            fps_start = now
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        if not args.no_display:
            cv2.imshow("People Count (current frame only)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()