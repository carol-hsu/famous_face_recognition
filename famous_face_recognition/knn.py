"""
Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.
"""

import math
from sklearn import neighbors
import pickle

def train(encodings, names, model_save_path=None, n_neighbors=None, knn_algo='ball_tree'):
    # Determine how many neighbors to use for weighting in the KNN classifier
    #if n_neighbors is None:
    #    n_neighbors = int(round(math.sqrt(len(X))))

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodings, names)
    
    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    
    return knn_clf


def predict(encodings, knn_clf=None, model_path=None, distance_threshold=0.6):

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    predicts = []
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(encodings, n_neighbors=1)
    preds = knn_clf.predict(encodings)
    
    predicts = [ preds[i] \
            if closest_distances[0][i][0] <= distance_threshold else "unknown" \
            for i in range(len(closest_distances[0])) ]


    return predicts #(pred[0], 1-closest_distances[0][0][0]) if is_match else ("unknown", -1)

