import cv2
import time
from datetime import datetime


first_frame=None
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vid=cv2.VideoCapture(0)
a=0
status_list=[0,0]
times=[]
while(True):
	check,frame=vid.read()
	a=a+1

	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	gray=cv2.GaussianBlur(gray,(21,21),0)
	status=0
	if(first_frame is None):
		first_frame=gray
		continue

	delta_frame=cv2.absdiff(first_frame,gray)
	thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
	#print(thresh_delta)
	
	thresh_delta=cv2.dilate(thresh_delta,None,iterations=3)

	(cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour)<100:
			continue
		status=1

		x,y,w,h=cv2.boundingRect(contour)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


	status_list.append(status)
	if status_list[-1]==1 and status_list[-2]==0:
		times.append(datetime.now())
	if status_list[-1]==0 and status_list[-2]==1:
		times.append(datetime.now())	
	cv2.namedWindow("Capture")
	cv2.imshow("Capture",gray)
	cv2.imshow("Delta",delta_frame)
	cv2.imshow("Threshold",thresh_delta)
	cv2.imshow("Color frame",frame)
	k=cv2.waitKey(1)

	if k==ord('q'):
		break
	
print(a)
print(status_list)


print(times)
vid.release()
cv2.destroyAllWindows()
