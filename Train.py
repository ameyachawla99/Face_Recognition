import cv2
import numpy as np
cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('face.xml')
skip=0
face_data=[]
dataset_path='./data/'
file_name=input('name of person')
for i in range(300):
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces,key=lambda f:f[2]*f[3])

	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
		offset=10
		face_section=frame[y+offset:y+h+offset,x+10:x+w+10]
		face_section=cv2.resize(face_section,(100,100))
		skip+=1
		if skip %10==0:
			face_data.append(face_section)

			print(len(face_data))
		cv2.imshow('face section',face_section)
	cv2.imshow('Video Capture',frame)

    

	cv2.waitKey(1)
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(dataset_path+file_name+'.npy',face_data)
print('data saved'+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()