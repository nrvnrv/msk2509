import numpy as np
import cv2
import joblib
from skimage.feature import hog


# Load the classifier
clf = joblib.load("digitreco/digits_cls.pkl")

# Default camera has index 0 and externally(USB) connected cameras have
# indexes ranging from 1 to 3
cap = cv2.VideoCapture(0)

while(True):


  # Capture frame-by-frame
  ret, frame = cap.read()

  # Convert to grayscale and apply Gaussian filtering
  im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)


  # Threshold the image
  ret, im_th = cv2.threshold(im_gray.copy(), 120, 255, cv2.THRESH_BINARY_INV)

  # Find contours in the binary image 'im_th'

  _, contours0, hierarchy  = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Draw contours in the original image 'im' with contours0 as input

  # cv2.drawContours(frame, contours0, -1, (0,0,255), 2, cv2.LINE_AA, hierarchy, abs(-1))


  # Rectangular bounding box around each number/contour
  rects = [cv2.boundingRect(ctr) for ctr in contours0]

  # Draw the bounding box around the numbers
  for rect in rects:

   cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

   # Make the rectangular region around the digit
   leng = int(rect[3] * 1.6)
   pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
   pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
   roi = im_th[pt1:pt1+leng, pt2:pt2+leng]



   # Resize the image
   roi = cv2.resize(roi, (28, 28), im_th, interpolation=cv2.INTER_AREA)
   roi = cv2.dilate(roi, (3, 3))
   # Calculate the HOG features
   roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
   nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
   cv2.putText(frame, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)



   # Display the resulting frame
   cv2.imshow('frame', frame)
   cv2.imshow('Threshold', im_th)



   # Press 'q' to exit the video stream
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()