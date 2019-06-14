import math
import numpy
import classifier_Gaze
import cv2
# ----------------------------------------------------------------------------------------------------------------------
def get_train_data(folder_in, file_annotations,target_W,target_H):

    with open(file_annotations) as f: lines = f.readlines()[1:]

    X,Y,Z=[],[],[]
    for line in lines:
        split = line.split('\n')[0]
        split = split.split('\t')
        image = cv2.imread(folder_in+split[0])
        if image is None:continue
        image = cv2.resize(image,(target_H,target_W))
        X.append(image)
        Y.append([split[2],split[3]])
        Z.append(split[4:])

    X = numpy.array(X)
    Y = numpy.array(Y,dtype=numpy.float)
    Z = numpy.array(Z)
    return X,Y,Z
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    folder_in = './data/input/24/'
    C = classifier_Gaze.classifier_Gaze()
    X_train,Y_train,Z_train = get_train_data(folder_in,folder_in+'markup_fact.txt',C.input_image_shape[0],C.input_image_shape[1])
    C.learn(X_train, Y_train,Z_train)
    C.save_model('./data/output/model.h5')

    #C.load_model('./data/output/model.h5')
    #Y_pred = C.predict([X_train,Z_train])
    #err = math.sqrt(numpy.mean((Y_pred - Y_train) ** 2))
    #print(err)



