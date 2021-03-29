import cv2
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('face.xml')

for i in range(300):
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(gray,1.3,5)
	
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,22),5)
	cv2.imshow('Video Capture',frame)

	cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()