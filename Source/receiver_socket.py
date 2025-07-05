import socket
import struct
import cv2
import pickle
import torch
import csv
import os
import time
import datetime
import threading
import face_recognition
import numpy as np
from pathlib import Path
from queue import Queue
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, cv2
from utils.torch_utils import select_device
from utils.plots import Annotator

# ------------------ 初始化 ------------------
weights = 'yolov5s.pt'
device = select_device('')
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = (640, 640)
model.warmup(imgsz=(1, 3, *imgsz))

video_save_dir = "saved_videos"
os.makedirs(video_save_dir, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

person_names = {1: "A", 2: "B", 3: "C"}


latest_frame1 = None
latest_frame2 = None
lock = threading.Lock()

def draw_infos_on_frame(frame, infos, client_id):
    if not infos:
        cv2.putText(frame, "Persons: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Moving Persons: 0", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return

    for idx, info in enumerate(infos):
        x1, y1, x2, y2 = info['x1'], info['y1'], info['x2'], info['y2']
        person_count = info['person_count']
        moving_person_count = info['moving_person_count']
        left, top, right, bottom = info['left'], info['top'], info['right'], info['bottom']
        name = info['name']

        if idx == len(infos) - 1:
            cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Moving Persons: {moving_person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        if name != 'Unknown':
            cv2.rectangle(frame, (left + x1, top + y1), (right + x1, bottom + y1), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + x1, top + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime()) + f"{int(time.time() * 1000) % 1000:03d}"
    filename = f"cutvideos/{client_id}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
def clean_old_videos(directory, days=7):
    now = datetime.datetime.now()
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
            if (now - creation_time).days > days:
                os.remove(file_path)
                print(f"已删除过期视频: {file_path}")

def get_latest_frame():
    with lock:
        return latest_frame1.copy() if latest_frame1 is not None else None
def get_latest_frame2():
    with lock:
        return latest_frame2.copy() if latest_frame2 is not None else None
def handle_client(conn, addr, client_id):
    global latest_frame1
    global latest_frame2
    print(f"[客户端 {client_id}] 连接自：{addr}")

    def get_video_path():
        client_dir = os.path.join(video_save_dir, client_id)
        os.makedirs(client_dir, exist_ok=True)  # 确保子文件夹存在
        filename = f"{client_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.mp4"
        return os.path.join(client_dir, filename)

    csv_path = Path(f"message/person_count_{client_id}.csv")
    label_csv_path = Path(f"message/detected_labels_{client_id}.csv")

    with open(csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow(["Frame", "Person Count", "Moving Person Count", "Person Name"])

    data = b""
    payload_size = struct.calcsize(">L")
    frame_id = 0
    current_minute = datetime.datetime.now().minute
    video_writer = None
    fps = 30
    frame_queue = Queue(maxsize=5)
    infos = []

    def process_frames():
        global latest_frame1
        global latest_frame2
        nonlocal frame_id, video_writer, current_minute
        prev_gray = None

        while True:
            if frame_queue.empty():
                continue

            frame = frame_queue.get()
            now_minute = datetime.datetime.now().minute
            if now_minute != current_minute or video_writer is None:
                if video_writer:
                    video_writer.release()
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(get_video_path(), cv2.VideoWriter_fourcc(*'mpv4'), fps, (w, h))
                current_minute = now_minute

            video_writer.write(frame)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)

            if prev_gray is None:
                prev_gray = curr_gray
                continue

            frame_delta = cv2.absdiff(prev_gray, curr_gray)
            motion_mask = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]

            im = cv2.resize(frame, imgsz)
            im_tensor = torch.from_numpy(im).to(device).permute(2, 0, 1).float() / 255.0
            im_tensor = im_tensor.unsqueeze(0)

            if frame_id % 3 == 0:
                pred = model(im_tensor, augment=False, visualize=False)
                pred = non_max_suppression(pred, 0.3, 0.45)

                annotator = Annotator(frame, line_width=2, example=str(names))
                person_count = 0
                moving_person_count = 0
                infos.clear()

                for det in pred:
                    if len(det):
                        det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in det:
                            if int(cls) != 0:
                                continue
                            x1, y1, x2, y2 = map(int, xyxy)
                            label = names[int(cls)]
                            # print(label)
                            if label == "person":
                                person_count += 1
                                person_mask = motion_mask[y1:y2, x1:x2]
                                area_threshold = (frame.shape[0] * frame.shape[1]) * 0.001
                                if cv2.countNonZero(person_mask) > area_threshold:
                                    moving_person_count += 1

                                if x2 - x1 <= 0 or y2 - y1 <= 0:
                                    continue

                                face_frame = frame[y1:y2, x1:x2]
                                h, w = face_frame.shape[:2]
                                blob = cv2.dnn.blobFromImage(face_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                                face_net.setInput(blob)
                                detections = face_net.forward()
                                gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)

                                name, left, top, right, bottom = "Unknown", 0, 0, 0, 0

                                for i in range(detections.shape[2]):
                                    confidence = detections[0, 0, i, 2]
                                    if confidence > 0.7:
                                        box = detections[0, 0, i, 3:7] * [w, h, w, h]
                                        left, top, right, bottom = box.astype('int')
                                        left, top = max(0, left), max(0, top)
                                        right, bottom = min(w, right), min(h, bottom)
                                        if right - left <= 0 or bottom - top <= 0:
                                            continue

                                        face_roi = gray[top:bottom, left:right]
                                        if face_roi.size == 0:
                                            continue

                                        face_roi = cv2.resize(face_roi, (200, 200))
                                        face_roi = cv2.equalizeHist(face_roi)

                                        id, conf = recognizer.predict(face_roi)
                                        name = person_names.get(id, "Unknown") if conf < 90 else "Unknown"

                                infos.append({
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'person_count': person_count,
                                    'moving_person_count': moving_person_count,
                                    'left': left, 'top': top, 'right': right, 'bottom': bottom,
                                    'name': name
                                })

                frame = annotator.result()

            draw_infos_on_frame(frame, infos, client_id)
            prev_gray = curr_gray.copy()
            disp = cv2.resize(frame, (640, 360))
            # print(client_id)
            with lock:
                if client_id=="client0":
                    latest_frame1 = disp.copy()
                if client_id=="client1":
                    latest_frame2 = disp.copy()

            frame_id += 1

        if video_writer:
            video_writer.release()
        try:
            cv2.destroyWindow(f"客户端 {client_id}")
        except:
            pass

    threading.Thread(target=process_frames, daemon=True).start()

    try:
        while True:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    raise ConnectionError("客户端断开连接")
                data += packet

            packed_size = data[:payload_size]
            data = data[payload_size:]
            frame_size = struct.unpack(">L", packed_size)[0]

            while len(data) < frame_size:
                packet = conn.recv(4096)
                if not packet:
                    raise ConnectionError("客户端断开连接")
                data += packet

            frame_data = data[:frame_size]
            data = data[frame_size:]

            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if not frame_queue.full():
                frame_queue.put(frame)

    except Exception as e:
        print(f"[客户端 {client_id}] 异常：{e}")
    finally:
        conn.close()
        print(f"[客户端 {client_id}] 断开连接")

def start_socket_server(host='0.0.0.0', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[监听] Socket 服务已启动：{host}:{port}")
    client_id_counter = 0
    clean_old_videos(video_save_dir)
    while True:
        conn, addr = server_socket.accept()
        client_id = f"client{client_id_counter}"
        client_id_counter += 1
        threading.Thread(target=handle_client, args=(conn, addr, client_id), daemon=True).start()
