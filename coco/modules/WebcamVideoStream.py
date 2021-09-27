from threading import Thread, Lock
import cv2


class WebcamVideoStream:
    def __init__(self, src=0):
        self.thread = Thread(target=self.update, daemon=True, args=())
        self.stream = cv2.VideoCapture(src)
        frame_width = int(self.stream.get(3))
        frame_height = int(self.stream.get(4))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE,0)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("There is an instance of WebcamVideoStream running already")
            return None
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()

if __name__=="__main__":
    source = 'blob:https://rtsp.me/af01d6b5-3667-4a96-a793-17f2efbc67a0'
    source = 0
    vs = WebcamVideoStream(source).start()
    while True :
        frame = vs.read()
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27 :
            break
    vs.stop()
    cv2.destroyAllWindows()