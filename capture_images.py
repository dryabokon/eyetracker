import cv2
import numpy
import tools_image
import PIL.ImageGrab
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import detector_landmarks
# ---------------------------------------------------------------------------------------------------------------------
def get_next_folder(folder_out):
    sub_folders = tools_IO.get_sub_folder_from_folder(folder_out)
    if len(sub_folders) > 0:
        sub_folders = numpy.array(sub_folders, dtype=numpy.int)
        sub_folders = numpy.sort(sub_folders)
        sub_folder_out = str(sub_folders[-1] + 1)
    else:
        sub_folder_out = '0'

    full_folder_out = folder_out + sub_folder_out + '/'
    #os.mkdir(full_folder_out)
    return full_folder_out
# ---------------------------------------------------------------------------------------------------------------------
def get_random_positions(i, minR, maxR, minC, maxC):

    freq1 = 10*2
    freq2 = 15*2

    r = minR + (maxR-minR)*(1+numpy.sin(i/freq1))/2
    c = minC + (maxC-minC)*(1+numpy.sin(i/freq2))/2
    r = numpy.minimum(numpy.maximum(minR, r), maxR)
    C = numpy.minimum(numpy.maximum(minC, c), maxC)

    return int(r),int(c)
# ---------------------------------------------------------------------------------------------------------------------
def get_image_calibration(W,H,r,c,R):
    image_calibration = numpy.full((W,H,3),128,dtype=numpy.uint8)
    cv2.circle(image_calibration, (int(W*(c/W)), int(H*(r/H))), R, [128, 255, 32], thickness=-1)
    return image_calibration
# ---------------------------------------------------------------------------------------------------------------------
def capture_images(folder_out):


    image_black = tools_image.get_screenshot()
    H,W,R = image_black.shape[0],image_black.shape[1],50
    D = detector_landmarks.detector_landmarks(H=H, W=W)

    tools_IO.remove_files(folder_out,create=True)

    cap = cv2.VideoCapture(1)
    cv2.namedWindow("frame")
    cv2.moveWindow("frame", -15, -14)

    show_face = False
    markups=[['filename_eyes','filename_screen','raw','col']+D.get_landmarks(None).tolist()]
    i = 0
    while True:

        r, c = get_random_positions(i, 2 * R, H - 2 * R, 2 * R, W - 2 * R)
        image_calibration = get_image_calibration(H,W,r,c,R)

        if show_face==False:
            cv2.imshow('frame', image_calibration)

        ret, frame = cap.read()
        frame = cv2.flip(frame,0)
        image_face = D.detect_face(frame)

        if image_face is not None:
            cv2.imwrite(folder_out + 'E_%04d.jpg' % (i), image_face)
            cv2.imwrite(folder_out + 'S_%04d.jpg' % (i), tools_image.get_screenshot())
            markup = ['E_%04d.jpg'%i,'S_%04d.jpg'%i, r/H, c/W]
            landmars = D.get_landmarks(frame)
            markups.append(markup + landmars.tolist())

        if show_face == True:
            frame = D.draw_face(frame)
            cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:break
        i+=1

    cap.release()
    tools_IO.save_mat(markups,folder_out+'markup_fact.txt')
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_out = './data/output/25/'
    capture_images(folder_out)


