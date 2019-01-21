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
#import os
#import os.path
import pickle
#from PIL import Image, ImageDraw

#ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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


def predict(encoding, knn_clf=None, model_path=None, distance_threshold=0.6):

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(encoding, n_neighbors=1)

    #TODO: currently only one face per image
    #are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    is_match = True if closest_distances[0][0][0] <= distance_threshold else False

    # Predict classes and remove classifications that aren't within the threshold
    #return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    pred = knn_clf.predict(encoding)
    return (pred[0], 1-closest_distances[0][0][0]) if is_match else ("unknown", -1)

'''
def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
'''
