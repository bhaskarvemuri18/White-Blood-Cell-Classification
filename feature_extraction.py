from skimage import io,transform
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import pandas as pd


def get_features(img_name):
    data = io.imread(img_name,as_grey=True)
    return data
def sift_features():
    data = get_features("/media/shashank/Study1/6140/project/dataset2-master/images/TEST_SIMPLE/EOSINOPHIL/_0_5239.jpeg")
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()

    keypoints, descriptors = sift.detectAndCompute(data, None)
    print keypoints[0]
    img = cv2.drawKeypoints(data, keypoints, None)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gray_scale_features(img_name):
    data = get_features(img_name)
    down_sample  = (60,80)
    img_read = transform.resize(
        data, down_sample, mode='constant')
    img_read = np.uint8(img_read*255)
    feat = np.ravel(img_read)

    return feat

#data = gray_scale_features("/media/shashank/Study1/6140/project/dataset2-master/images/TEST_SIMPLE/EOSINOPHIL/_0_5239.jpeg")
#print data.shape

#im = Image.fromarray(data)
#im.show()

def write_features_to_csv(file_name):
    paths = ["/media/shashank/Study1/6140/project/dataset2-master/images/TEST/EOSINOPHIL/","/media/shashank/Study1/6140/project/dataset2-master/images/TEST/LYMPHOCYTE/","/media/shashank/Study1/6140/project/dataset2-master/images/TEST/MONOCYTE/","/media/shashank/Study1/6140/project/dataset2-master/images/TEST/NEUTROPHIL/"]
    k = 1
    for classs in range(len(paths)):
        mypath = paths[classs]
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for fl in onlyfiles:
            if '.jpeg' in fl:
                print k
                feat = gray_scale_features(mypath + fl)
                feat = np.append(feat,classs)
                out = []
                out.append(feat)
                out = np.array(out)
                if k == 1:
                    pd.DataFrame(out).to_csv(file_name,mode='a')
                else:
                    pd.DataFrame(out).to_csv(file_name,mode='a',header=False)
                k+=1

write_features_to_csv("/media/shashank/Study1/6140/project/dataset2-master/testing_data.csv")
