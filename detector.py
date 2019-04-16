import cv2
import numpy as np
import serial
from flask import Flask, render_template, Response
from camera import VideoCamera
import time
arduinodata=serial.Serial('com17',9600)
app = Flask(__name__)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainningData.yml")
id=0
count=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while(True):
    ret,img=cam.read();
    if ret:#warm up the camera
       gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
       faces=faceDetect.detectMultiScale(gray,1.3,5);
       for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            print(id)
            if(id==1):
                id="Admin"
                #arduinodata.write(faces)
            
            else:
             id="intruder"
             count=count+1
                
             if(count==10):
                arduinodata.write("1")
                arduinodata.close()
                cam.release()
                cv2.destroyAllWindows()
                @app.route('/')
                def index():
                    return render_template('index.html')
                def gen(camera):
                  while True:
                    frame = camera.get_frame()
                    yield (b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                @app.route('/video_feed')
                def video_feed():
                    return Response(gen(VideoCamera()),
                                    mimetype='multipart/x-mixed-replace; boundary=frame')
                if __name__ == '__main__':
                       app.run(host='0.0.0.0', debug=True)
                
                    
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
       cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
