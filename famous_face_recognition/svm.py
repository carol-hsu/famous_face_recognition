import cv2, os
from numpy import *
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA#, RandomizedPCA
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
#import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imutils import paths

IMAGE_ROW=140
IMAGE_COL=140
#THE_ONE = "natalie_portman"
#THE_OTHER = "keira_knightley"


def prepare_dataset(directory):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cascadeLocation = dir_path+"/haarcascade_frontalface_default.xml" ## need to download in local?
    faceCascade = cv2.CascadeClassifier(cascadeLocation)

    image_paths = list(paths.list_images(directory))
    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = image_path.split('/')[-2]
        #grab faces
        faces = faceCascade.detectMultiScale(image) 
        for (x, y, w, h) in faces:
            images.append(image[y:y+IMAGE_COL, x:x+IMAGE_ROW])
            labels.append(nbr)
#            cv2.imshow("Reading Faces ",image[y:y+IMAGE_COL, x:x+IMAGE_ROW])
#            cv2.waitKey(50)
    
    return images, labels

def train(faces, labels, svm_kernal="rbf"):

    n_components = 10
    cv2.destroyAllWindows()
    #pca = RandomizedPCA(n_components=n_components, whiten=True) #n components for aggrating as a "thing"? (unsupervised
    pca = PCA(n_components=n_components, whiten=True,svd_solver='randomized') #n components for aggrating as a "thing"? (unsupervised
    
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier = GridSearchCV(SVC(kernel=svm_kernal), param_grid) #class_weight='balanced in SVC
    
    training_data = []
    training_label = []
    for i in range(len(faces)):
        temp = faces[i].flatten()
        if len(temp) == IMAGE_ROW*IMAGE_COL:
            training_data.append(temp)
            training_label.append(labels[i])
        
    #print len(faces)
    pca = pca.fit(training_data)

    transformed = pca.transform(training_data) #transform the training set and put it in SVM
    classifier.fit(transformed, training_label) 

    return pca, classifier

def predict(faces, pca, clf):
    # prediction
#        pred_image_pil = Image.open(image_path).convert('L')
#        pred_image = np.array(pred_image_pil, 'uint8')
#        faces = faceCascade.detectMultiScale(pred_image)
    predicts = []
    for face in faces:
        temp = np.array(face).flatten()
        if len(temp) == IMAGE_ROW*IMAGE_COL:
            X_test = pca.transform([np.array(face).flatten()])
            mynbr = clf.predict(X_test)
            predicts.append(mynbr[0])
        else:
            predicts.append("unknown")

    return predicts

