from flask import Flask, Response, render_template, send_from_directory, url_for
import threading
import cv2
import os
from receiver_socket import start_socket_server, get_latest_frame, get_latest_frame2

app = Flask(__name__)
VIDEO_DIR = "saved_videos"

os.makedirs(VIDEO_DIR, exist_ok=True)

# MJPEG 视频流生成器
def generate_mjpeg():
    while True:
        frame = get_latest_frame()
        if frame is None:
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
def generate_mjpeg2():
    while True:
        frame = get_latest_frame2()
        if frame is None:
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
# sdjh
@app.route('/fullscreen.html')
def fullscreen():
    return render_template(
        'fullscreen.html',
        video1=url_for('video1'),
        video2=url_for('video2')
    )


@app.route('/video1')
def video1():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video2')
def video2():
    return Response(generate_mjpeg2(), mimetype='multipart/x-mixed-replace; boundary=frame')
# ✅ 视频设备分类展示页面
@app.route('/videos')
def video_list():
    device_list = []
    for device_name in os.listdir(VIDEO_DIR):
        device_path = os.path.join(VIDEO_DIR, device_name)
        if os.path.isdir(device_path):
            videos = []
            for filename in os.listdir(device_path):
                if filename.endswith('.mp4'):
                    videos.append({
                        'name': filename,
                        'url': url_for('serve_device_video', device=device_name, filename=filename)
                    })
            videos.sort(key=lambda x: x['name'], reverse=True)
            device_list.append({
                'device': device_name,
                'videos': videos
            })
    return render_template('videos.html', device_list=device_list)

# ✅ 单个视频播放（指定设备）
@app.route('/videos/<device>/<filename>')
def serve_device_video(device, filename):
    return send_from_directory(os.path.join(VIDEO_DIR, device), filename)

if __name__ == '__main__':
    threading.Thread(target=start_socket_server, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
