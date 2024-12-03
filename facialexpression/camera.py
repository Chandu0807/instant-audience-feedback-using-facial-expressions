
import threading

import cv2
import numpy as np
import tensorflow as tf

from facialexpression.expressionmodel import predict, image_to_tensor, deepnn
import dlib
import imutils
import os

from facialexpression.models import FeedBackModel, CurrentFeedback

PROJECT_PATH = os.path.abspath(os.path.dirname(__name__))

detector = dlib.get_frontal_face_detector()

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def format_image(image):

  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]

  images=[]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
      # face to image
      face_coor =  max_are_face
      image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
      # Resize image to network size
      try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
        images.append(image)
      except Exception as e:
        print(e)
        print("[+} Problem during resize")
        return None, None
  return  images

def face_dect(image):
  """
  Detecting faces in image
  :param image:
  :return:  the coordinate of max face
  """
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  if not len(faces) > 0:
    return None
  max_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_face[2] * max_face[3]:
      max_face = face
  face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
  try:
    image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
  except Exception as e:
    print(e)
    print("[+} Problem during resize")
    return None
  return face_image

def resize_image(image, size):
  try:
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("+} Problem during resize")
    return None
  return image

def draw_emotion():
  pass

face_x = tf.placeholder(tf.float32, [None, 2304])
y_conv = deepnn(face_x)
probs = tf.nn.softmax(y_conv)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(PROJECT_PATH+"/ckpt")
sess = tf.Session()
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')
    print(ckpt)

facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_dict = {"angry":0,"disgusted":0,"fearful":0,"happy":0,"sad":0,"surprised":0,"neutral":0}
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

class VideoCamera(object):
    
    def start(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def stop(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(),image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera,request):

    i=0
    while True:

        frame,image = camera.get_frame()

        if i==500:
            i=0

            if image is not None:
                image = imutils.resize(image, width=450)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # detect faces in the grayscale frame
                faces = detector(gray, 0)

                if len(faces) > 0:
                    print("No of faces : ", len(faces))
                    for face in faces:
                        x = face.left()
                        y = face.top()  # could be face.bottom() - not sure
                        w = face.right() - face.left()
                        h = face.bottom() - face.top()
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if len(faces) > 0:

                    for face in faces:

                        x = face.left()
                        y = face.top()  # could be face.bottom() - not sure
                        w = face.right() - face.left()
                        h = face.bottom() - face.top()

                        roi_gray = gray[y:y + h, x:x + w]  # croping
                        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                        tensor = image_to_tensor(cropped_img)
                        result = sess.run(probs, feed_dict={face_x: tensor})

                        maxindex = int(np.argmax(result[0]))
                        label = EMOTIONS[maxindex]
                        print("Label:", label)

                        emotion_dict[label] = emotion_dict[label] + 1

                        for key, value in emotion_dict.items():
                            print(key, ":", value)

                        cv2.waitKey(1)

                    total = 0
                    for key, value in emotion_dict.items():
                        total = total + value

                    if total != 0:
                        dict1 = {"angry": emotion_dict['angry'] / total,
                                 "disgusted": emotion_dict['disgusted'] / total,
                                 "fearful": emotion_dict['fearful'] / total,
                                 "happy": emotion_dict['happy'] / total,
                                 "sad": emotion_dict['sad'] / total,
                                 "surprised": emotion_dict['surprised'] / total,
                                 "neutral": emotion_dict['neutral'] / total}

                        if dict1['angry'] > dict1['disgusted'] \
                                and dict1['angry'] > dict1['fearful'] \
                                and dict1['angry'] > dict1['happy'] \
                                and dict1['angry'] > dict1['sad'] \
                                and dict1['angry'] > dict1['surprised'] \
                                and dict1['angry'] > dict1['neutral']:
                            cv2.putText(image, "more angry", (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                                        1)

                        elif dict1['disgusted'] > dict1['angry'] \
                                and dict1['disgusted'] > dict1['fearful'] \
                                and dict1['disgusted'] > dict1['happy'] \
                                and dict1['disgusted'] > dict1['sad'] \
                                and dict1['disgusted'] > dict1['surprised'] \
                                and dict1['disgusted'] > dict1['neutral']:
                            cv2.putText(image, "more disgusted", (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 255, 255), 1, 1)

                        elif dict1['fearful'] > dict1['angry'] \
                                and dict1['fearful'] > dict1['disgusted'] \
                                and dict1['fearful'] > dict1['happy'] \
                                and dict1['fearful'] > dict1['sad'] \
                                and dict1['fearful'] > dict1['surprised'] \
                                and dict1['fearful'] > dict1['neutral']:
                            cv2.putText(image, "more fearful", (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                                        1, 1)

                        elif dict1['happy'] > dict1['angry'] \
                                and dict1['happy'] > dict1['disgusted'] \
                                and dict1['happy'] > dict1['fearful'] \
                                and dict1['happy'] > dict1['sad'] \
                                and dict1['happy'] > dict1['surprised'] \
                                and dict1['happy'] > dict1['neutral']:
                            cv2.putText(image, "more happy", (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                                        1)

                        elif dict1['sad'] > dict1['angry'] \
                                and dict1['sad'] > dict1['disgusted'] \
                                and dict1['sad'] > dict1['fearful'] \
                                and dict1['sad'] > dict1['happy'] \
                                and dict1['sad'] > dict1['surprised'] \
                                and dict1['sad'] > dict1['neutral']:
                            cv2.putText(image, "more sad", (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                                        1)

                        elif dict1['surprised'] > dict1['angry'] \
                                and dict1['surprised'] > dict1['disgusted'] \
                                and dict1['surprised'] > dict1['fearful'] \
                                and dict1['surprised'] > dict1['happy'] \
                                and dict1['surprised'] > dict1['sad'] \
                                and dict1['surprised'] > dict1['neutral']:
                            cv2.putText(image, "more surprised", (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 255, 255), 1, 1)

                        elif dict1['neutral'] > dict1['angry'] \
                                and dict1['neutral'] > dict1['disgusted'] \
                                and dict1['neutral'] > dict1['fearful'] \
                                and dict1['neutral'] > dict1['happy'] \
                                and dict1['neutral'] > dict1['sad'] \
                                and dict1['neutral'] > dict1['surprised']:
                            cv2.putText(image, "more neutral", (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                                        1, 1)

                        pcount = (dict1['neutral'] + dict1['surprised'] + dict1['happy'])
                        ncount = (dict1['angry'] + dict1['sad'] + dict1['disgusted'] + dict1['fearful'])

                        CurrentFeedback.objects.all().update(pcount=pcount,ncount=ncount)
                        
                        print("Result",pcount,ncount)

            print("reached..")
        else:
            i=i+1
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')