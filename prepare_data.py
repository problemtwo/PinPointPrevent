import os
import sys
import cv2

cas = cv2.CascadeClassifier('face_cascade.xml')

def main(d,output_dir):
	data = [cv2.imread(os.path.join(d,fl)) for fl in os.listdir(d)]
	i = 0
	for k in data:
		faces = cas.detectMultiScale(k,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))
		for (x,y,w,h) in faces:
			print('Writing image training/{}/{}.jpg'.format(output_dir,i))
			i += 1
			cv2.imwrite(os.path.join('training',output_dir,str(i)+'.jpg'),k[y:y+h,x:x+w])

if __name__ == '__main__' and len(sys.argv) > 1:
	main(sys.argv[1],sys.argv[2])
