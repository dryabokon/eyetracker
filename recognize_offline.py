import os
import cv2
import numpy
import math
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import classifier_Gaze
import tools_image
import tools_animation
# ---------------------------------------------------------------------------------------------------------------------
def process_annotation_file(C,folder_in, file_annotations,file_result):

    with open(file_annotations) as f: lines = f.readlines()[1:]
    result=[['filename_eyes','fliename_screenshot','raw','col']]

    for line in lines:
        split = line.split('\n')[0]
        split = split.split('\t')
        image = cv2.imread(folder_in+split[0])
        if image is None:continue
        X = cv2.resize(image,(C.input_image_shape[1],C.input_image_shape[0]))
        X = numpy.array([X])
        Z = numpy.array([split[4:]])
        res = C.predict([X, Z])[0]
        result.append([split[0],split[1],res[0],res[1]])



    result = numpy.array(result)
    tools_IO.save_mat(result,file_result)

    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_markup(folder_in, file_annotations_fact,file_annotations_pred,folder_out):

    image_black = tools_image.get_screenshot()
    H, W = image_black.shape[0], image_black.shape[1]

    tools_IO.remove_files(folder_out, create=True)

    with open(file_annotations_fact) as f: lines_fact = f.readlines()[1:]
    with open(file_annotations_pred) as f: lines_pred = f.readlines()[1:]

    for line in lines_pred:
        split = line.split('\n')[0]
        split = split.split('\t')
        image = cv2.imread(folder_in+split[1])
        if image is None:continue

        r,c,R = split[2],split[3],10
        r = int(float(r)*H)
        c = int(float(c)*W)
        r = min(1080-2*R, max(float(r), 2*R))
        c = min(1920-2*R, max(float(c), 2*R))
        cv2.circle(image, (int(c), int(r)), R, [255, 64,0], thickness=-1)
        cv2.imwrite(folder_out+split[1],image)
    print('Pred OK')


    for line in lines_fact:
        split = line.split('\n')[0]
        split = split.split('\t')
        image = cv2.imread(folder_out+split[1])
        if image is None:continue

        r,c,R = split[2],split[3],int(10)
        r = int(float(r)*H)
        c = int(float(c)*W)
        r = min(H-2*R, max(float(r), 2*R))
        c = min(W-2*R, max(float(c), 2*R))

        image = tools_image.desaturate(image)
        cv2.circle(image, (int(c), int(r)), R, [32, 190, 0], thickness=-1)
        cv2.imwrite(folder_out+split[1],image)
    print('Fact OK')


    return
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_accuracy(file_annotations_fact,file_annotations_pred):

    image_black = tools_image.get_screenshot()
    H, W = image_black.shape[0], image_black.shape[1]

    with open(file_annotations_fact) as f: lines_fact = f.readlines()[1:]
    with open(file_annotations_pred) as f: lines_pred = f.readlines()[1:]

    dct_r,dct_c={},{}
    error = []
    Y_pred = []
    Y_fact = []

    for line in lines_fact:
        split = line.split('\n')[0]
        split = split.split('\t')
        r,c = float(split[2]),float(split[3])
        #r = int(float(r)*H)
        #c = int(float(c)*W)
        dct_r[split[0]]=r
        dct_c[split[0]]=c
        Y_fact.append([r,c])


    for line in lines_pred:
        split = line.split('\n')[0]
        split = split.split('\t')
        r,c = float(split[2]),float(split[3])
        #r = int(float(r)*H)
        #c = int(float(c)*W)

        dist = math.sqrt((dct_c[split[0]]-c)**2 +(dct_r[split[0]]-r)**2)
        error.append(dist)
        Y_pred.append([r, c])

    error = numpy.mean(error)

    #Y_pred = numpy.array(Y_pred)
    #Y_fact = numpy.array(Y_fact)
    #err = math.sqrt(numpy.mean((Y_pred - Y_fact) ** 2))

    return error
# ----------------------------------------------------------------------------------------------------------------------
def visualize_annotations(folder_in, file_annotations,folder_out,color=[32, 190, 0]):

    if not os.path.exists(folder_out):os.mkdir(folder_out)

    with open(file_annotations) as f: lines = f.readlines()[1:]

    for line in lines:
        split = line.split('\n')[0]
        split = split.split('\t')
        image = cv2.imread(folder_in + split[0])
        if image is None: continue
        r,c = float(split[2]),float(split[3])
        r = int(float(r)*image.shape[0])
        c = int(float(c)*image.shape[1])
        cv2.circle(image, (int(c), int(r)), 5, color, thickness=-1)
        cv2.imwrite(folder_out + split[0], image)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    C = classifier_Gaze.classifier_Gaze()
    folder_in  = './data/input/23/'
    folder_out = './data/output/23a/'

    C.load_model('./data/output/model.h5')
    process_annotation_file(C,folder_in,folder_in+'markup_fact.txt',folder_in+'markup_pred.txt')
    #draw_markup(folder_in, folder_in+'markup_fact.txt',folder_in+'markup_pred.txt','./data/output/25a/')
    #error = evaluate_accuracy(folder_in+'markup_fact.txt',folder_in+'markup_pred.txt')
    #print(error)

    visualize_annotations(folder_in,  folder_in + 'markup_fact.txt',folder_out)
    visualize_annotations(folder_out, folder_in + 'markup_pred.txt',folder_out,color=[255, 64,0])
    tools_animation.folder_to_animated_gif_imageio(folder_out, folder_out+'animation.gif', mask='*.jpg', framerate=10,resize_H=64, resize_W=64)