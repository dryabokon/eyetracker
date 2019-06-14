import numpy
import cv2
import dlib
import tools_image
# --------------------------------------------------------------------------------------------------------------------
class detector_landmarks(object):
    def __init__(self,H=1080,W=1920,mode='dlib'):
        self.W=W
        self.H=H
        self.mode = mode
        self.name = "landmark_detector"
        self.the_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

# --------------------------------------------------------------------------------------------------------------------
    def detect_face(self,image):
        if self.mode=='opencv':
            return self.__detect_face_opencv(image)
        else:
            return self.__detect_face_dlib(image)

# --------------------------------------------------------------------------------------------------------------------
    def draw_face(self, image):
        if self.mode == 'opencv':
            return self.__draw_face_opencv(image)
        else:
            return self.__draw_face_dlib(image)
# --------------------------------------------------------------------------------------------------------------------
    def __detect_face_dlib(self,image):
        gray = tools_image.desaturate(image)
        objects = self.detector(gray)
        if len(objects) == 1:
            for face in objects:
                xmin = face.left()
                ymin = face.top()
                xmax = face.right()
                ymax = face.bottom()
                return image[ymin:ymax, xmin:xmax, :]

        return None
# ----------------------------------------------------------------------------------------------------------------------
    def __detect_face_opencv(self, image):
        objects = self.the_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(objects) == 2:
            xmin = min(objects[0][0], objects[1][0])
            xmax = max(objects[0][0] + objects[0][2], objects[1][0] + objects[1][2])

            ymin = min(objects[0][1], objects[1][1])
            ymax = max(objects[0][1] + objects[0][3], objects[1][1] + objects[1][3])

            if xmax > xmin and ymax > ymin:
                return image[ymin:ymax, xmin:xmax, :]

        return None
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_face_opencv(self, image):
        objects = self.the_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(objects) == 2:
            for (x, y, w, h) in objects: cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def __draw_face_dlib(self,image):
        gray = tools_image.desaturate(image)
        objects = self.detector(gray)
        if len(objects) == 1:
            for face in objects:
                xmin = face.left()
                ymin = face.top()
                xmax = face.right()
                ymax = face.bottom()
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def get_landmarks(self,image):
        res = numpy.zeros(68 * 2, dtype=numpy.float)
        if image is None:
            return res
        gray = tools_image.desaturate(image)
        objects = self.detector(gray)

        if len(objects) == 1:
            landmarks = self.predictor(gray, objects[0])
            res=[]
            for n in range(0, 68):
                res.append(landmarks.part(n).x/self.W)
                res.append(landmarks.part(n).y/self.H)
            res = numpy.array(res)

        return res
# ----------------------------------------------------------------------------------------------------------------------