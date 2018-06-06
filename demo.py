import cv2
import os
import time

cas = cv2.CascadeClassifier('face_cascade.xml')

def main():
	cap = cv2.VideoCapture(0)
	dt = time.time()

	name = raw_input('What is your name? ')
	if not os.path.exists(os.path.join('training',name)):
		os.mkdir(os.path.join('training',name))

	num_faces = 0

	while True:
		if round(time.time() - dt) >= 30:
			return
		_,im = cap.read()
		faces = cas.detectMultiScale(im,scaleFactor=1.2,minNeighbors=5)
		if len(faces) > 0:
			(x,y,w,h) = faces[0]
			num_faces += 1
			cv2.imwrite(os.path.join('.','training',name,str(num_faces)+'.jpg'),im[y:y+h,x:x+w])
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
			cv2.putText(im,'Writing image training/{}/{}'.format(name,num_faces),(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
		cv2.imshow('Demo',im)
		k = cv2.waitKey(30)
		if k == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	name = main()
