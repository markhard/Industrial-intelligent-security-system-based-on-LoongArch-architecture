import cv2 as cv  # 导入OpenCV库，通常用于处理图像和视频
from threading import Thread  # 导入线程模块，用于创建和管理线程
import time  # 导入时间模块，用于计时


class CamWidget:
    def __init__(self, path):
        """
        初始化方法，创建一个CamWidget对象。

        参数:
            path (str): 视频文件或摄像头的路径。
        """
        # 使用给定的路径创建VideoCapture对象，用于读取视频或摄像头流
        self.cap = cv.VideoCapture(path)

        # 设置视频流的分辨率为640x480
        self.cap.set(3, 640)  # 设置宽度
        self.cap.set(4, 480)  # 设置高度

        # 设置视频流的曝光和其他属性
        self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)  # 开启自动曝光
        self.cap.set(cv.CAP_PROP_EXPOSURE, 150)  # 设置曝光值
        self.cap.set(cv.CAP_PROP_BRIGHTNESS, 30)  # 设置亮度

        # 尝试设置视频的帧率，但注意不是所有摄像头都支持设置帧率
        self.cap.set(cv.CAP_PROP_FPS, 30)

        # 一个标志，用于在读取图像的线程中判断是否应释放摄像头
        self.thread_release = False

        # 保存视频或摄像头的路径
        self.path = path

        # 用于存储从摄像头或视频中读取的当前帧图像
        self.image = None

        # 创建一个守护线程，用于在后台读取图像
        self.thread = Thread(target=self.read_image, args=())
        self.thread.daemon = True  # 设置为守护线程，确保主程序退出时该线程也退出
        self.thread.start()  # 启动线程

        # 等待直到读取到第一帧图像
        while self.image is None:
            pass

    def read_image(self):
        """
        后台线程函数，用于不断从摄像头或视频中读取图像。
        """
        while True:
            if not self.thread_release:
                # 尝试从VideoCapture对象中读取一帧图像
                ret, image = self.cap.read()
                if not ret:
                    # 如果读取失败，则释放当前的VideoCapture对象并重新打开
                    self.cap.release()
                    print("read_video_error")
                    self.cap = cv.VideoCapture(self.path)

                    # 等待直到VideoCapture对象重新打开成功
                    t1 = time.time()
                    while not self.cap.isOpened():
                        if time.time() - t1 > 2:
                            # 如果等待超过2秒仍未打开，则再次尝试重新打开
                            self.cap.release()
                            self.cap = cv.VideoCapture(self.path)
                            t1 = time.time()
                else:
                    # 如果读取成功，则更新self.image变量为读取到的图像
                    self.image = image
            else:
                # 如果收到释放摄像头的信号，则释放VideoCapture对象并退出循环
                self.cap.release()
                break

    def close(self):
        """
        关闭方法，用于释放摄像头资源并等待后台线程结束。
        """
        # 设置释放摄像头的标志
        self.thread_release = True
        # 等待后台线程结束
        self.thread.join()
