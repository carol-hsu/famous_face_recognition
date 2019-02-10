# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
from imutils import paths
from famous_face_recognition import knn, decision_tree, svm, neural_network

DETECTION_METHOD="cnn"

def old_method(encodings, names, face_image):

    for face in face_image:
    # attempt to match each face in the input image to our known
	# encodings
        matches = face_recognition.compare_faces(encodings, face)
        confidence = face_recognition.face_distance(encodings, face)
        name = "unknown"
    
        # check to see if we have found a match
        if True in matches:
        # find the indexes of all matched faces then initialize a dictionary 
        # to count the total number of times each face was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            confid_value = {}
        
        # loop over the matched indexes and maintain a count for
        # each recognized face face
            for i in matchedIdxs:
                name = names[i]
                counts[name] = counts.get(name, 0) + 1
                if name in confid_value:
                    confid_value[name].append(confidence[i])
                else:
                    confid_value[name] = [confidence[i]]
            
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
                name = max(counts, key=counts.get)
    
    # update the list of names
	## names.append(name)
    return (name, 1-min(confid_value[name]))


def encode_image(image_file):
    # load the input image and convert it from BGR to RGB
    image = cv2.imread(image_file)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)

    return encodings


def load_encodings(encoding_file):
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    f = open(encoding_file, "rb")
    
    data = {}
    
    while True:
        try:
            if not data:
                data = pickle.load(f)
            else: 
                temp_data = pickle.load(f)
                data["encodings"] = data["encodings"] + temp_data["encodings"]
                data["names"] = data["names"] + temp_data["names"]
    
        except EOFError as e:
            #bad... but workable
            print("[INFO] Finish loading! Get "+str(len(data["names"]))+" faces ")
            break

    return data

def accuracy(predicts, answers):
    correct = len(answers)
    for i in range(len(answers)):
        if predicts[i] != answers[i]:
            correct-=1
    print correct/float(len(answers))*100

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=False,
            help="directory of training set/encodings of training set")
    ap.add_argument("-i", "--test", required=False,
            help="directory of testing set/encodings of testing set")
    ap.add_argument("-v", "--validate", required=False,
            help="directory of validating set/encoding of validating set")
    ap.add_argument("-m", "--learning-model", type=int, default=5,
            help="Use which model for learning: 1=knn, 2=decision tree, 3=boosting, 4=svm, 5=neural network")
    ap.add_argument("-n", "--number-neighbors", type=int, default=8,
            help="neighbor number for knn model (default: 8)")
    ap.add_argument("-k", "--kernel", type=str, default="rbf",
            help="kernel for svm model, linear/poly/rbf/sigmoid/precomputed (default: rbf)")
    ap.add_argument("-b", "--boosting-estimators", type=int, default=10,
            help="the number of estimators for boosting")
    params = vars(ap.parse_args())
    
    #trained + inference + test    
    model_type = params["learning_model"]

    
    if model_type < 4: #these algos use 128 features encodings
        #load encodings
        faces = load_encodings(params["train"])
        test_faces = load_encodings(params["test"])
        
        if model_type == 1: ### knn
            print("[INFO] Apply knn model with k = "+str(params["number_neighbors"]))
            knn_clfier = knn.train(faces["encodings"], faces["names"], n_neighbors=params["number_neighbors"])
            predicts = knn.predict([enc.tolist() for enc in test_faces["encodings"]], knn_clf=knn_clfier)
            accuracy(predicts, test_faces["names"])

        elif model_type == 2:
            print("[INFO] Apply decision tree method...")
            tree = decision_tree.build_tree(faces["encodings"], faces["names"])
            decision_tree.tree_info(tree)

            pruned_tree = decision_tree.prune(tree) 
            decision_tree.tree_info(pruned_tree)
            print pruned_tree.score([enc.tolist() for enc in test_faces["encodings"]], test_faces["names"])*100

        else: ### boosting
            print("[INFO] Apply boosting method...")
            boosted_tree = decision_tree.boosting(faces["encodings"], faces["names"], \
                                                  estimators=params["boosting_estimators"])
            print boosted_tree.score([enc.tolist() for enc in test_faces["encodings"]], test_faces["names"])*100


    elif model_type == 4: 
        #read images from scratchs
        print("[INFO] Apply SVM...")
        images, labels = svm.prepare_dataset(params["train"])
        pca, clf = svm.train(images, labels, svm_kernal=params["kernel"])
        test_images, test_labels = svm.prepare_dataset(params["test"])

        predicts = svm.predict(test_images, pca, clf)
        accuracy(predicts, test_labels)

    else: #model_type = 5

        print("[INFO] Apply neural network...")
        model = neural_network.build_model()    
        database = neural_network.train(params["train"], model)
        predicts = neural_network.predict(params["test"], database, model)
        accuracy(predicts, \
                [path.split("/")[-2] for path in list(paths.list_images(params["test"]))])


