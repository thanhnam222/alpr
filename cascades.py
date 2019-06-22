import cv2 as cv
cars = cv.CascadeClassifier('cascade.xml')
img = cv.imread('/home/nam/Desktop/folder/alpr-unconstrained-master/samples/test/03016.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

car = cars.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in car:
	cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
	roi_gray = gray[y:y + h, x:x + w]
	print('a')
	roi_color = img[y:y + h, x:x + w]
	cars = cars.detectMultiScale(roi_gray)
	for (ex, ey, ew, eh) in car:
		cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()