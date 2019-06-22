# import cv2
# import numpy as np
# import os
# from keras.models import model_from_json
#
# labelsPath = "/home/nam/Desktop/yolo-object-detection/yolo-coco/coco.names"
# LABELS = open(labelsPath).read().strip().split("\n")
#
# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
# 	dtype="uint8")
#
# weightsPath = "/home/nam/Desktop/yolo-object-detection/yolo-coco/yolov3-320.weights"
# configPath = "/home/nam/Desktop/yolo-object-detection/yolo-coco/yolov3-320.cfg"
# print("[INFO] loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#
# path_images = "/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/cars_train"
#
# idx = 0
# idx1 = 0
# for root, dir, file in os.walk(path_images):
# 	idx1 += 1
# 	if idx1 > 1000:
# 		break
# 	for f in file:
# 		image = cv2.imread(root + "/" + f)
# 		(H, W) = image.shape[:2]
# 		ln = net.getLayerNames()
# 		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# 		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
# 		net.setInput(blob)
#
# 		layerOutputs = net.forward(ln)
#
# 		boxes = []
# 		confidences = []
# 		classIDs = []
#
# 		cars = []
#
# 		for output in layerOutputs:
# 			# loop over each of the detections
# 			for detection in output:
# 				# extract the class ID and confidence (i.e., probability) of
# 				# the current object detection
# 				scores = detection[5:]
# 				classID = np.argmax(scores)
# 				confidence = scores[classID]
#
# 				if LABELS[classID] != "car" and LABELS[classID] != "bus" and LABELS[classID] != "truck":
# 					continue
# 				# filter out weak predictions by ensuring the detected
# 				# probability is greater than the minimum probability
# 				if confidence > 0.5:
# 					# scale the bounding box coordinates back relative to the
# 					# size of the image, keeping in mind that YOLO actually
# 					# returns the center (x, y)-coordinates of the bounding
# 					# box followed by the boxes' width and height
# 					box = detection[0:4] * np.array([W, H, W, H])
# 					(centerX, centerY, width, height) = box.astype("int")
#
# 					# use the center (x, y)-coordinates to derive the top and
# 					# and left corner of the bounding box
# 					x = int(centerX - (width / 2))
# 					y = int(centerY - (height / 2))
#
# 					# update our list of bounding box coordinates, confidences,
# 					# and class IDs
# 					boxes.append([x, y, int(width), int(height)])
# 					confidences.append(float(confidence))
# 					classIDs.append(classID)
#
# 					im = image[y:y+int(height), x:x+int(width)]
# 					if(np.array(im).shape[0]<1 or np.array(im).shape[1]<1):
# 						continue
# 					# cv2.imshow("g", im)
# 					# cv2.waitKey(0)
#
# 					cv2.imwrite("/home/nam/Desktop/folder/data/Cars/" + "a" + str(idx) + ".jpg" , im)
# 					idx += 1
# 					# cars.append(im)
#
# 		# apply non-maxima suppression to suppress weak, overlapping bounding
# 		# boxes
# 		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
#
# 		# ensure at least one detection exists
# 		if len(idxs) > 0:
# 			# loop over the indexes we are keeping
# 			for i in idxs.flatten():
# 				# extract the bounding box coordinates
# 				(x, y) = (boxes[i][0], boxes[i][1])
# 				(w, h) = (boxes[i][2], boxes[i][3])
#
# 				# draw a bounding box rectangle and label on the image
# 				color = [int(c) for c in COLORS[classIDs[i]]]
# 				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
# 				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
# 				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
# 					0.5, color, 2)
#
# 		# show the output image
# 		# cv2.imshow("Image", image)
# 		# cv2.waitKey(0)
#
#
# class Label:
#
# 	def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
# 		self.__tl = tl
# 		self.__br = br
# 		self.__cl = cl
# 		self.__prob = prob
#
# 	def __str__(self):
# 		return 'Class: %d, top_left(x:%f,y:%f), bottom_right(x:%f,y:%f)' % (
# 		self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])
#
# 	def copy(self):
# 		return Label(self.__cl, self.__tl, self.__br)
#
# 	def wh(self): return self.__br - self.__tl
#
# 	def cc(self): return self.__tl + self.wh() / 2
#
# 	def tl(self): return self.__tl
#
# 	def br(self): return self.__br
#
# 	def tr(self): return np.array([self.__br[0], self.__tl[1]])
#
# 	def bl(self): return np.array([self.__tl[0], self.__br[1]])
#
# 	def cl(self): return self.__cl
#
# 	def area(self): return np.prod(self.wh())
#
# 	def prob(self): return self.__prob
#
# 	def set_class(self, cl):
# 		self.__cl = cl
#
# 	def set_tl(self, tl):
# 		self.__tl = tl
#
# 	def set_br(self, br):
# 		self.__br = br
#
# 	def set_wh(self, wh):
# 		cc = self.cc()
# 		self.__tl = cc - .5 * wh
# 		self.__br = cc + .5 * wh
#
# 	def set_prob(self, prob):
# 		self.__prob = prob
#
#
#
# def getWH(shape):
# 	return np.array(shape[1::-1]).astype(float)
# class DLabel (Label):
#
# 	def __init__(self,cl,pts,prob):
# 		self.pts = pts
# 		tl = np.amin(pts,1)
# 		br = np.amax(pts,1)
# 		Label.__init__(self,cl,tl,br,prob)
#
#
# def IOU(tl1, br1, tl2, br2):
# 	wh1, wh2 = br1 - tl1, br2 - tl2
# 	assert ((wh1 >= .0).all() and (wh2 >= .0).all())
#
# 	intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)
# 	intersection_area = np.prod(intersection_wh)
# 	area1, area2 = (np.prod(wh1), np.prod(wh2))
# 	union_area = area1 + area2 - intersection_area;
# 	return intersection_area / union_area
#
#
# def IOU_labels(l1,l2):
# 	return IOU(l1.tl(),l1.br(),l2.tl(),l2.br())
# def nms(Labels, iou_threshold=.5):
# 	SelectedLabels = []
# 	Labels.sort(key=lambda l: l.prob(), reverse=True)
#
# 	for label in Labels:
#
# 		non_overlap = True
# 		for sel_label in SelectedLabels:
# 			if IOU_labels(label, sel_label) > iou_threshold:
# 				non_overlap = False
# 				break
#
# 		if non_overlap:
# 			SelectedLabels.append(label)
#
# 	return SelectedLabels
# def getRectPts(tlx,tly,brx,bry):
# 	return np.matrix([[tlx,brx,brx,tlx],[tly,tly,bry,bry],[1.,1.,1.,1.]],dtype=float)
#
#
# def find_T_matrix(pts, t_pts):
# 	A = np.zeros((8, 9))
# 	for i in range(0, 4):
# 		xi = pts[:, i];
# 		xil = t_pts[:, i];
# 		xi = xi.T
#
# 		A[i * 2, 3:6] = -xil[2] * xi
# 		A[i * 2, 6:] = xil[1] * xi
# 		A[i * 2 + 1, :3] = xil[2] * xi
# 		A[i * 2 + 1, 6:] = -xil[0] * xi
#
# 	[U, S, V] = np.linalg.svd(A)
# 	H = V[-1, :].reshape((3, 3))
#
# 	return H
#
#
# def reconstruct(Iorig,I,Y,out_size,threshold=.9):
#
# 	net_stride 	= 2**4
# 	side 		= ((208. + 40.)/2.)/net_stride # 7.75
#
# 	Probs = Y[...,0]
# 	Affines = Y[...,2:]
# 	rx,ry = Y.shape[:2]
# 	ywh = Y.shape[1::-1]
# 	iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))
#
# 	xx,yy = np.where(Probs>threshold)
#
# 	WH = getWH(I.shape)
# 	MN = WH/net_stride
#
# 	vxx = vyy = 0.5 #alpha
#
# 	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
# 	labels = []
#
# 	for i in range(len(xx)):
# 		y,x = xx[i],yy[i]
# 		affine = Affines[y,x]
# 		prob = Probs[y,x]
#
# 		mn = np.array([float(x) + .5,float(y) + .5])
#
# 		A = np.reshape(affine,(2,3))
# 		A[0,0] = max(A[0,0],0.)
# 		A[1,1] = max(A[1,1],0.)
#
# 		pts = np.array(A*base(vxx,vyy)) #*alpha
# 		pts_MN_center_mn = pts*side
# 		pts_MN = pts_MN_center_mn + mn.reshape((2,1))
#
# 		pts_prop = pts_MN/MN.reshape((2,1))
#
# 		labels.append(DLabel(0,pts_prop,prob))
#
# 	final_labels = nms(labels,.1)
# 	TLps = []
#
# 	if len(final_labels):
# 		final_labels.sort(key=lambda x: x.prob(), reverse=True)
# 		for i,label in enumerate(final_labels):
#
# 			t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
# 			ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
# 			H 		= find_T_matrix(ptsh,t_ptsh)
# 			Ilp 	= cv2.warpPerspective(Iorig,H,out_size,borderValue=.0)
#
# 			TLps.append(Ilp)
#
# 	return final_labels,TLps
#
#
# def detect_lp(model,I,max_dim,net_step,out_size,threshold):
#
# 	min_dim_img = min(I.shape[:2])
# 	factor 		= float(max_dim)/min_dim_img
#
# 	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
# 	w += (w%net_step!=0)*(net_step - w%net_step)
# 	h += (h%net_step!=0)*(net_step - h%net_step)
# 	Iresized = cv2.resize(I,(w,h))
#
# 	T = Iresized.copy()
# 	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))
#
# 	# start 	= time.time()
# 	Yr 		= model.predict(T)
# 	Yr 		= np.squeeze(Yr)
# 	elapsed = 0
#
# 	L,TLps = reconstruct(I,Iresized,Yr,out_size,threshold)
#
# 	return L,TLps,elapsed
# def im2single(I):
# 	assert(I.dtype == 'uint8')
# 	return I.astype('float32')/255.
#
#
# # json_file = open('/home/nam/Desktop/folder/alpr-unconstrained-master/data/lp-detector/wpod-net_update1.json', 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()
# # modell = model_from_json(loaded_model_json)
# # modell.load_weights("/home/nam/Desktop/folder/alpr-unconstrained-master/data/lp-detector/wpod-net_update1.h5")
#
# # idx = 0
# # for car in cars:
# # 	Ivehicle = car
# # 	ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
# # 	side = int(ratio * 288.)
# # 	bound_dim = min(side + (side % (2 ** 4)), 608)
# #
# # 	Llp, LlpImgs, _ = detect_lp(modell, im2single(Ivehicle), bound_dim, 2 ** 4, (240, 80), 0.5)
# #
# # 	print(len(LlpImgs))
# # 	if (len(LlpImgs)):
# # 		for i in LlpImgs:
# # 			# cv2.imshow("12", i)
# # 			# cv2.waitKey(0)
# # 			i = i * 255
# # 			cv2.imwrite("/home/nam/Desktop/folder/data/pl/" + str(idx) + ".jpg", i)
# # 			print("write", idx)
# # 			idx += 1
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

import cv2
import numpy as np

f = open("/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/training-dataset-annotations/cars-dataset/00790.txt", "r")
txt = f.read().split(",")
print(txt)
img = cv2.imread("/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/training-dataset-annotations/cars-dataset/00790.jpg")
(h, w, _) = np.array(img).shape
print(w, h)

# tl = (int(float(txt[1]) * w), int(float(txt[5]) * h))
# br = (int(float(txt[3]) * w), int(float(txt[7]) * h))
# img = cv2.rectangle(img, tl, br, (0,255,0), 3)

img = cv2.circle(img,(int(float(txt[1]) * w),int(float(txt[5]) * h)), 5, (0,0,255), -1)
img = cv2.circle(img,(int(float(txt[2]) * w),int(float(txt[6]) * h)), 5, (0,0,255), -1)
img = cv2.circle(img,(int(float(txt[3]) * w),int(float(txt[7]) * h)), 5, (0,0,255), -1)
img = cv2.circle(img,(int(float(txt[4]) * w),int(float(txt[8]) * h)), 5, (0,0,255), -1)
#
# img = cv2.circle(img,(int(float(txt[11]) * w),int(float(txt[15]) * h)), 3, (0,0,255), -1)
# img = cv2.circle(img,(int(float(txt[12]) * w),int(float(txt[16]) * h)), 3, (0,0,255), -1)
# img = cv2.circle(img,(int(float(txt[13]) * w),int(float(txt[17]) * h)), 3, (0,0,255), -1)
# img = cv2.circle(img,(int(float(txt[14]) * w),int(float(txt[18]) * h)), 3, (0,0,255), -1)

cv2.imshow("a", img)
cv2.waitKey(0)