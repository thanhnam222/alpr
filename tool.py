import cv2
import numpy as np
import os


# txt = []
# for root, dir, file in os.walk("/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/training-dataset-annotations/cars-dataset"):
# 	for f in file:
# 		f = f.split(".")[0]
# 		txt.append(f + ".jpg")
# print(txt)
# for root, dir, file in os.walk("/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/cars_train"):
# 	for f in file:
# 		f = f.split(".")[0] + ".jpg"
# 		if f in txt:
# 			img = cv2.imread("/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/cars_train/" + f)
# 			cv2.imwrite("/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/training-dataset-annotations/cars-dataset/" + f, img)


# for root, dir, files in os.walk("/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/training-dataset-annotations/cars-dataset"):
# 	for file in files:
# 		fi = file.split(".")[-1]
# 		if(fi != "txt"): continue
# 		print(file)
# 		with open("/home/nam/Desktop/folder/alpr-unconstrained-master/newFolder/training-dataset-annotations/cars-dataset/" + file,"r+") as f:
# 			new_f = f.readline()
# 			# print(new_f)
# 			new_f = new_f.split(",")
# 			f.seek(0)
# 			for i in range(9):
# 				f.write(new_f[i] + ",")
# 			f.write(",")
# 			f.truncate()

import cv2
import numpy as np
import imutils

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

toaDo = []

def draw_circle(event,x,y,flags,param):
	global ix, iy, drawing, mode, toaDo
	if event == cv2.EVENT_LBUTTONDOWN:
		# print(x, y)
		toaDo.append((x, y))
		# print("toaDo: ", toaDo)
		drawing = True
		ix, iy = x, y
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == False:
				cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
			else:
				cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == False:
			cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
		else:
			cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


for root, dir, files in os.walk("/home/nam/Desktop/tool"):
	for file in files:
		toaDo = []
		print(file)
		img = cv2.imread("/home/nam/Desktop/tool/" + file)
		if(img.shape[1] > 1200):
			img = imutils.resize(img, width = 1200)
		if(img.shape[0] > 800):
			r = img.shape[1] / img.shape[0] * 650
			img = imutils.resize(img, width = int(r))
		cv2.namedWindow("image")
		cv2.setMouseCallback("image",draw_circle)
		h, w, _ = img.shape
		while(1):
			cv2.imshow('image',img)
			k = cv2.waitKey(1) & 0xFF
			# if k == ord('m'):
			# 	mode = not mode
			if k == 27:
				break

		cv2.destroyAllWindows()

		idx = 0
		td = []
		TD = []
		for i in toaDo:
			TD.append(i)
			idx += 1
			if(idx %4 == 0):
				td.append(TD)
				TD = []
		with open("/home/nam/Desktop/tool/" + file.split(".")[0] + ".txt", 'w') as f:
			for t in td:
				(x1, y1), (x2, y2), (x3, y3), (x4, y4) = t
				x1 = round(x1/w, 6)
				x2 = round(x2/w, 6)
				x3 = round(x3/w, 6)
				x4 = round(x4/w, 6)
				y1 = round(y1/h, 6)
				y2 = round(y2/h, 6)
				y3 = round(y3/h, 6)
				y4 = round(y4/h, 6)
				f.write("4," + str(x1) +","+ str(x2) +","+ str(x3) +","+  str(x4) +","+ str(y1) +","+ str(y2) +","+ str(y3) +","+ str(y4)+",,")
				f.write("\n")

