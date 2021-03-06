import sys
import cv2 as cv
import time
from .WebcamVideoStream import WebcamVideoStream
from threading import Thread
import os.path


source = 0 # 'http://94.72.19.56/mjpg/video.mjpg'
objnames = 'coco/obj.names'
cfgfile = 'coco/yolov4-tiny.cfg'
weightfile = 'coco/yolov4-tiny.weights'


class Detect:
    def __init__(self, net_size=0):
        print("initialization start")
        self.thread = Thread(target=self.my_frame_producer, daemon=True, args=())
        self.started = False
        self.enabled = False
        with open(objnames, 'rt') as f:
            self.class_names = f.read().rstrip('\n').split('\n')
        self.net = cv.dnn_DetectionModel(cfgfile,weightfile)
        self.net.setInputSize(net_size, net_size)  # 416, 416 for better accuracy, set in run.py
        self.net.setInputScale(1.0 / 127)  # 1.0 / 256 for better accuracy
        self.net.setInputSwapRB(True)
        print("initialization done")
        
        
    def my_frame_producer(self):
        print("Start video recieving")
        stream = WebcamVideoStream(source).start()
        print("Receiving video")
        while True:
            frame = stream.read()
            classes, confidences, boxes = self.net.detect(frame, confThreshold=0.3, nmsThreshold=0.4)
            if len(boxes) > 0:
                for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                    label = '%.2f' % confidence
                    label = '%s: %s' % (self.class_names[classId], label)
                    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    left, top, width, height = box
                    top = max(top, label_size[1])
                    cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                    cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                                 (255, 255, 255),
                                 cv.FILLED)
                    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            frame = cv.resize(frame, (1200, 700))
            cv.imshow('Detect', frame)

            if cv.waitKey(27) == 27:
                cv.destroyAllWindows()
                self.enabled = False
                sys.exit(0)


    def detect_data(self):
        while self.started:
            while self.enabled:
                while True:
                    start_time = time.time()
                    frame = self.vs.read()
                    classes, confidences, boxes = self.net.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
                    if len(boxes) > 0:
                        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                            label = '%.2f' % confidence
                            label = '%s: %s' % (self.class_names[classId], label)
                            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            left, top, width, height = box
                            top = max(top, label_size[1])
                            cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                            cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                                         (255, 255, 255),
                                         cv.FILLED)
                            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    frame = cv.resize(frame, (1200, 700))
                    cv.imshow('Detect', frame)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    if cv.waitKey(27) == 27:
                        cv.destroyAllWindows()
                        self.enabled = False
                        sys.exit(0)


    def start(self):
        print("Start")
        if self.started:
            print("There is an instance of Detect running already")
            return None
        self.started = True
        self.thread.start()
        return self


    def stop(self):
        self.started = False
        self.thread.join()


    def enable(self):
        self.enabled = True


    def disable(self):
        self.enabled = False
