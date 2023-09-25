import cv2 as cv
import dlib
import numpy as np
import datetime
from functools import partial

window_name = "Blink Detector"

class PredictorBlinkClass:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.right_eye = (36, 42)
        self.left_eye = (42, 48)
        self.threshold = 0.17
        self.blinks = 0
        self.crossed_right = False
        self.crossed_left = False
        self.time_crossed = 1
        self.time_start = 1
        self.blink_time = 1000
        self.prop = 1.15
        self.bpm = 0
        # Button
        self.button_x = 140
        self.button_y = 420
        self.button_width = 200
        self.button_height = 50
    
    def rect_to_ndarray(self, shape, length):
        coords = np.zeros((length, 2), dtype="int")
        for i in range(0, length):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def distance(self, p1, p2):
        temp = p1 - p2
        return np.sqrt(np.dot(temp.T, temp))

    def get_ear(self, eye):
        a = self.distance(eye[1], eye[5])
        b = self.distance(eye[2], eye[4])
        c = self.distance(eye[0], eye[3])
        return (a + b) / (2 * c)

    def set_threshold(self, threshold):
        self.threshold = round(0.100 + (threshold * 1.0) / 100.0, 2)

    def set_blinks(self, blinks):
        self.blinks = blinks

    def reset(self):
        self.blinks = 0
        self.time_crossed = 0
        self.crossed_right = False
        self.crossed_left = False
        self.time_start = 0
        self.bpm = 0

    def predict(self, image_color, time, RGB=(0, 0, 255)):
        image_predicted = image_color.copy()
        image_gray = cv.cvtColor(image_color, cv.COLOR_BGR2GRAY)
        faceboxes = self.detector(image_gray, 1)
        ear_right = 0.0
        ear_left = 0.0
        elapsed_time = time - self.time_start
        for facebox in faceboxes:
            if self.time_crossed > 1 and time - self.time_crossed > self.blink_time:
                break

            if self.time_start == 0:
                self.time_start = time
            
            predicted = self.predictor(image_gray, facebox)
            landmarks = self.rect_to_ndarray(predicted, 68)
            
            # eye coordinates
            right, left = landmarks[self.right_eye[0]:self.right_eye[1]], landmarks[self.left_eye[0]:self.left_eye[1]]
            # get ear for each eye
            ear_right, ear_left = self.get_ear(right), self.get_ear(left)

            if (ear_right < self.threshold or ear_left < self.threshold):
                self.crossed_right = True
                self.crossed_left = True
                self.time_crossed = time
            
            if (self.crossed_right == True and ear_right > self.threshold * self.prop) or (self.crossed_left == True and ear_left > self.threshold * self.prop):
                self.crossed_right = False
                self.crossed_left = False
                self.time_crossed = 1
                self.blinks += 1
            
            for eye in [right, left]:
                for (x, y) in eye:
                    cv.circle(image_predicted, (x, y), 2, RGB, -1)
                    cv.circle(image_predicted, (x, y), 2, RGB, -1)

        if elapsed_time < 60000:
            self.bpm = self.blinks
        else:
            self.bpm = round(self.blinks/(elapsed_time/60000), 2)
        cv.putText(image_predicted, f"Blinks: {self.blinks}", (20, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        cv.putText(image_predicted, f"BlinksXMin: {self.bpm}", (20, 90), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        cv.putText(image_predicted, f"RightEAR: {round(ear_right, 3)}", (360, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        cv.putText(image_predicted, f"LeftEAR:  {round(ear_left, 3)}", (360, 90), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        cv.putText(image_predicted, f"Threshold: {self.threshold}", (20, 400), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        cv.putText(image_predicted, f"Time(s): {round(elapsed_time/1000, 1)}", (330, 400), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        cv.rectangle(image_predicted,
                     (self.button_x, self.button_y),
                     (self.button_x + self.button_width, self.button_y + self.button_height),
                     (0, 0, 255),
                     -1)
        cv.putText(image_predicted, f"RESET", (180, 450), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
        return image_predicted
    
def update_threshold(value):
    predictor.set_threshold(value)

def handle_mouse_event(button_x, button_y, button_width, button_height, event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # Check if the mouse click is within the button rectangle
        if button_x < x < button_x + button_width and button_y < y < button_y + button_height:
            predictor.reset()

def show_feed(feed, title, wait=1, rewind=False):
    retval, frame_color = feed.read()
    if not retval:
        return 1 # check value did not pass
    
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.setWindowProperty(title, cv.WND_PROP_VISIBLE, 1)
    cv.createTrackbar("Threshold", title, 7, 15, update_threshold)

    # Button
    button_x = 140
    button_y = 420
    button_width = 200
    button_height = 50
    
    custom_callback = partial(handle_mouse_event, button_x, button_y, button_width, button_height)
    cv.setMouseCallback(title, custom_callback)

    # Time
    time = datetime.datetime.now()
    
    while retval:
        time_delta = datetime.datetime.now() - time
        time_delta = int(time_delta.total_seconds() * 1000)
        predicted = predictor.predict(frame_color, time_delta)
        cv.rectangle(predicted, (button_x, button_y), (button_x + button_width, button_y + button_height), (0, 0, 255), 2)
        cv.imshow(title, predicted)
        
        key_input = cv.waitKey(wait)
        if key_input != -1:
            break
        
        retval, frame_color = feed.read()
    
    try:
        if rewind: # Rewind VideoCapture object for video files
            feed.set(cv.CAP_PROP_POS_MSEC, 0.0)
        cv.destroyWindow(title)
        
    except:
        print("'" + title + "' not found")

predictor = PredictorBlinkClass()

webcam_index = 0
webcam = cv.VideoCapture(webcam_index)

show_feed(webcam, window_name, wait=1, rewind=True)

webcam.release()