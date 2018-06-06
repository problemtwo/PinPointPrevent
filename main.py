import cv2 # OpenCV 2 version 3.4.1
import dlib # dlib is a c++ library which we are using to detect the front of a face (with dlib.get_frontal_face_detector())
import numpy as np # numpy is a library which opencv uses for it's images, so it is useful for conversion
import math # used for sine, cosine, and pi
import sys # used for sys.argv
import os # used to fiddle with the filesystem

pred = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detect = dlib.get_frontal_face_detector()
rec = cv2.face.LBPHFaceRecognizer_create(threshold=95)
cas = cv2.CascadeClassifier('face_cascade.xml')

def det(gray_image,save):
	store = []
	for (x,y,w,h) in cas.detectMultiScale(gray_image,1.3,6):
		save += [(x,y,w,h)]
		f = cv2.resize(gray_image[y:y+h,x:x+w],(100,100))
		store += [f]
	return store

# Borrowed from http://pranavdheer.co/face-recognition-a-step-by-step-guide/
def align(save):
	output = [] 
	flag = 0                                   #check if we entered loop
	for ix in range(0,len(save)): 
		flag=0
		detections = detect(save[ix],2)
		for k,d in enumerate(detections):
			shape = pred(save[ix], d)   #68 facial points
			p1 = [(shape.part(45).x,shape.part(45).y),(shape.part(36).x,shape.part(36).y)]
			p2 = [((int(0.7*100),33)),(int(0.3*100),33)]
			s60 = math.sin(60*math.pi/180)
			c60 = math.cos(60*math.pi/180) 
			inPts = np.copy(p1).tolist()
			outPts = np.copy(p2).tolist()
			xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
			yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
			inPts.append([np.int(xin), np.int(yin)])
			xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
			yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
			outPts.append([np.int(xout), np.int(yout)])
			tform = cv2.estimateRigidTransform(np.array([[int(j) for j in i] for i in inPts]), np.array([[int(p) for p in o] for o in outPts]), False)
			img2 = cv2.warpAffine(save[ix], tform, (100,100));

			detections = detect(img2,3)
			for k,d in enumerate(detections): 
				flag = 1
				face = [[abs(d.left()),abs(d.right())],[abs(d.top()),abs(d.bottom())]]
				shape = pred(img2, d)
				l_eye = np.asarray([(shape.part(36).x,shape.part(36).y),(shape.part(37).x,shape.part(37).y),(shape.part(38).x,shape.part(38).y),(shape.part(39).x,shape.part(39).y),(shape.part(40).x,shape.part(40).y),(shape.part(41).x,shape.part(41).y)])
				r_eye = np.asarray([(shape.part(42).x,shape.part(42).y),(shape.part(43).x,shape.part(43).y),(shape.part(44).x,shape.part(44).y),(shape.part(45).x,shape.part(45).y),(shape.part(46).x,shape.part(46).y),(shape.part(47).x,shape.part(47).y)])
				eye_left = np.mean(l_eye,axis=0)
				eye_right = np.mean(r_eye,axis=0) 
				face[0][0] = int((eye_left[0]+face[0][0])/2.0)
				face[0][1] = int((eye_right[0]+face[0][1])/2.0)
				face[1][1] = int((shape.part(10).y+shape.part(57).y)/2.0)
				img2_cropped = img2[face[1][0]:face[1][1],face[0][0]:face[0][1]]
				img2_cropped = cv2.resize(img2_cropped,(100,100))
				output.append(img2_cropped) 
			if flag == 0:
				del(save[ix])    #delete face coordinates with improper alignment
	if len(output) == 0: 
		return ('n',1) 
	return ('y',output)

def train(name): 
	data = [cv2.resize(
	         cv2.cvtColor(
           cv2.imread(os.path.join('training',name,flname)),cv2.COLOR_BGR2GRAY),(100,100))
            for flname in os.listdir(os.path.join('training',name)) if flname.split('.')[-1] in ['JPG','jpg','PNG','png']]
	images = []
	for im in data:
		save = []
		save += [im]
		output = align(save)
		if output[0] == 'y':
			images += [output[1][0]]
	rec.train(np.asarray(images),np.asarray([1 for i in images]))


def main(cmd_input=False,img=None,path=sys.argv[1],tr=True,outp=True):
	if tr:
		train(path)
	faces_save = []    
	save = []
	if cmd_input == False:
		cap = cv2.VideoCapture(0)   #reading video
		frame_width = int(cap.get(3))         
		frame_height = int(cap.get(4))
		frame_rate = 24
		frame_count = 0               #frame number, that we are reading
		while(cap.isOpened()):      #while video is still open
			ret, frame = cap.read()
			frame_count = frame_count+1
			(dimensions_x,dimensions_y,z) = frame.shape 
			images = frame 
			copy_image = images
			images = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)  #convert into grayscale
			detected = []
			if frame_count >= frame_rate or frame_count == 1:   #condition is valid once in a frame rate
				save = []                                  
				faces_save = det(images,save)
				output = align(faces_save)
				frame_count = 1                              #Again wait for 24 more frames to process the video
			if output[0]=='y':
				for ix in range (0,len(output[1])):
					x,y,w,h=save[ix][0],save[ix][1],save[ix][2],save[ix][3]
					out = output[1][ix]
					(iden,conf) = rec.predict(out)    #Predict label of face
					if iden == 1:
						detected += [(x,y,w,h,conf)]
					else:
						cv2.rectangle(copy_image,(x,y),(x+w,y+h), (0, 0, 255), 2)
			if len(detected) > 0:
				detected.sort(key=lambda x: x[4])
				(x,y,w,h,c) = detected[0]
				cv2.rectangle(copy_image,(x,y),(x+w,y+h), (0, 255, 0), 2)
				cv2.putText(copy_image,'Name: {}'.format(path),(x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
				cv2.putText(copy_image,'Confidence: {}/100'.format(round(c * 100) / 100),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
			
			cv2.imshow('video',copy_image)
			key = cv2.waitKey(30)
			if key == 27:
				break
		cap.release()
		cv2.destroyAllWindows()
	else:
		(dimensions_x,dimensions_y,z) = img.shape 
		images = img 
		copy_image = images
		images = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)  #convert into grayscale
		detected = []
		faces_save = det(images,save)
		output = align(faces_save)
		if(output[1] != 1):
			retrn = []
			for ix in range (0,len(output[1])):
				x,y,w,h=save[ix][0],save[ix][1],save[ix][2],save[ix][3]
				out = output[1][ix]
				lab = rec.predict(out)    #Predict label of face
				if lab[0]==1 and outp:
					print('Found {}, confidence: {}, [x:{},y:{},w:{},h:{}]'.format(path,lab[1],x,y,w,h))
				elif outp:
					print('Suspicious individual detected at [x:{},y:{},w:{},h:{}]'.format(x,y,w,h))
					#cv2.rectangle(copy_image,(x,y),(x+w,y+h), (0, 0, 255), 2)
				retrn += [[x,y,w,h,lab[0],lab[1]]]
			return retrn		

def multiple():
	res = []
	for i in os.listdir('training'):
		train(i)
		x = main(cmd_input=True,img=cv2.imread(sys.argv[2]),tr=False,outp=False)
		if x != None:
			for j in x:
				j.append(sys.argv[3])
			res += x
	res.sort(key=lambda y:y[5])
	#print(res)
	print('Found {}, confidence: {}, [x:{},y:{},w:{},h:{}]'.format(res[0][-1],res[0][-2],res[0][0],res[0][1],res[0][2],res[0][3]))

if __name__ == '__main__':
	if len(sys.argv) > 4:
		if sys.argv[3] == '-i':
			# python3 main.py [path-to-training-images-inside-training-directory] [num-training-images] -i [path-to-test-image]
			# ex. python3 main.py Abhi 10 -i training/Abhi/1.jpg
			main(cmd_input=True,tr=True,outp=True,img=cv2.imread(sys.argv[4]))
	elif sys.argv[1] == '-m':
		multiple()
	else:
		main()
