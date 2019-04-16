import cv2
import numpy as np

rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainningData.yml")
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        
        success, image = self.video.read()
        if success:
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces=faceDetect.detectMultiScale(gray,1.3,5);
            for(x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                id,conf=rec.predict(gray[y:y+h,x:x+w])
                if(id==3):
                    id="Admin"
                else:
                    id="intruder"
                    
                cv2.cv.PutText(cv2.cv.fromarray(image),str(id),(x,y+h),font,255);
  
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
