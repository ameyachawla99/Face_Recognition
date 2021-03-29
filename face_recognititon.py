import numpy as np
import cv2
import os
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

# Test Time 
def knn(X,Y,queryPoint,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    
    vals = np.array(vals)
    
    #print(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
	    
    return pred
cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('face.xml')
skip=0
face_data=[]
dataset_path='./data/'
labels=[]
class_id=0
names={}
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)
		target=class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0)
print(face_dataset.shape)
print(face_labels.shape)
for i in range(300):
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	
	for (x,y,w,h) in faces:
		offset=10
		face_section=frame[y+offset:y+h+offset,x+10:x+w+10]
		face_section=cv2.resize(face_section,(100,100))
		out=knn(face_dataset,face_labels,face_section.flatten())
		pred=names[int(out)]
		cv2.putText(frame,pred,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
	cv2.imshow('Video Capture',frame)
	cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()





