import cv2
import numpy
import tools_image
# ---------------------------------------------------------------------------------------------------------------------
import detector_landmarks
import classifier_Gaze
# ---------------------------------------------------------------------------------------------------------------------
def capture_and_recognize(C):
    image_black = tools_image.get_screenshot()
    image_black[:,:,:]=128
    image_res = image_black.copy()
    H, W = image_black.shape[0], image_black.shape[1]
    D = detector_landmarks.detector_landmarks(H=H, W=W)
    cap = cv2.VideoCapture(1)

    while (True):
        ret, frame = cap.read()
        image_face = D.detect_face(frame)


        if image_face is not None:
            X = cv2.resize(image_face, (C.input_image_shape[1], C.input_image_shape[0]))
            X = numpy.array([X])
            Z = numpy.array([D.get_landmarks(frame)])
            res = C.predict([X, Z])[0]

            r, c, R = res[0], res[1], int(10)
            r = int(float(r) * H)
            c = int(float(c) * W)
            r = min(H - 2 * R, max(float(r), 2 * R))
            c = min(W - 2 * R, max(float(c), 2 * R))

            image_res = image_black.copy()
            cv2.circle(image_res, (int(c), int(r)), R, [32, 190, 0], thickness=-1)

        cv2.imshow('frame', image_res)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:break

    cap.release()

    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    C = classifier_Gaze.classifier_Gaze()
    C.load_model('./data/output/model.h5')
    capture_and_recognize(C)
