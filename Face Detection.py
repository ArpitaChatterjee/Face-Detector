import cv2
from random import randrange

from google.colab.patches import cv2_imshow

#load some pretrained frontal face frm opencv (haar cascade algorithm)
#its supervised lerning where the algo has 5 haar features.
#step1-> distinguish face n non face
#step2-> weve to test every haar feature... on training img- every type , size location(each hf gives a no right or wrong)
# which every haar features chooses the training img closest is our first winner -- its decides by the color differences in the img
#so we change the img into grayscale so that its easy for the hf to detect the light n dark areas in the face(eyes darker than cheeks) n deteact the exact location of the face.
#step3-> the 'cascade'-- once we got 1st 1000 winner of hf, we can chain them together into our face dectector which is donr by cascadeclassifier n stored in xml file



trained_facedata = cv2.CascadeClassifier('/content/drive/MyDrive/Colab Notebooks/Face-Detector/haarcascade_frontalface_default.xml')

#choose the img to be read
#img = cv2.imread('/content/drive/MyDrive/Colab Notebooks/Face-Detector/rdj.jpg')
img = cv2.imread('/content/drive/MyDrive/Colab Notebooks/Face-Detector/bts3.jpg')
#convert it to greyscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_img)
cv2_imshow(img)
cv2.waitKey()


#convert it to greyscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_img)
cv2.waitKey()


print(face_coordinates)
#upperleft corner(144,128) and downright corner(297, 297) coordinates
#(x, y, w, h) x,y-coord & width, height of the img


#draw rectectangles (x,y), (x+w, y+h), (b,g,r), widthofrectangle
for (x,y, w, h) in face_coordinates:
  cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(255),randrange(255),randrange(255)), 10)

cv2_imshow(img)
cv2.waitKey()


#VIDEO VERSION

webcam = cv2.VideoCapture('/content/drive/MyDrive/Colab Notebooks/Face-Detector/gif.mp4')

#iterate over the frame
while True:
  ##read current frame
  successful_frame_read, frame= webcam.read()

  #must convert to greyscale
  gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  face_coordinates = trained_facedata.detectMultiScale(gray_img)

  for (x,y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(255),randrange(255),randrange(255)), 10)

  cv2_imshow(frame)
  key= cv2.waitKey(1) 

  if key == 81 or key==113:
    break
#clean the data
webcam.release()