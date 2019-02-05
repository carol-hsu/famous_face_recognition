# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
from famous_face_recognition import knn, decision_tree, svm, neural_network

DETECTION_METHOD="cnn"

def old_method(encodings, names, face_image):
    # initialize the list of names for each face detected 
    ##TODO: currently only one face per image, do not consider a multiple face issues
    ##names = []
    
    ## update encodings with the algos
    # loop over the facial embeddings
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
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)

    return encodings, boxes


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

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=False,
            help="path to serialized db of facial encodings")
    ap.add_argument("-t", "--train", required=False,
            help="directory of training images")
    ap.add_argument("-v", "--validate-encodings", required=False,
            help="path to serialized db of facial encodings, for validation")
    ap.add_argument("-i", "--image", required=False,
            help="directory of testing images")
    ap.add_argument("-m", "--learning-model", type=int, default=0,
            help="Use which model for learning: 0=none, 1=decision tree, 2=boost 3=knn, 4=svm, 5=neural network")
    ap.add_argument("-n", "--number-neighbors", type=int, default=8,
            help="neighbor number for knn model (default: 8)")
    ap.add_argument("-k", "--kernel", type=str, default="rbf",
            help="kernel for svm model, linear/poly/rbf/sigmoid/precomputed (default: rbf)")
    params = vars(ap.parse_args())
    
    pred = ("unknown", -1)

    #trained + inference + test    
    model_type = params["learning_model"]

    
    if model_type < 4:
        #load encodings
        faces = load_encodings(params["encodings"])
        input_enc, input_loc = encode_image(params["image"])

        if model_type == 3: ### knn
            print("[INFO] Apply knn model with k = "+str(params["number_neighbors"]))
            knn_clfier = knn.train(faces["encodings"], faces["names"], n_neighbors=params["number_neighbors"])
            pred = knn.predict(input_enc, knn_clf=knn_clfier)
        else: 
            print("[INFO] Apply decision tree method...")
            tree = decision_tree.train(faces["encodings"], faces["names"])
            if model_type == 2: ### boost
                #Grab data from validation encodings
                vfaces = load_encodings(params["validate_encodings"])
                #Do Pruning
                decision_tree.boost(vfaces["encodings"], vfaces["names"], tree) 

            decision_tree.predict(input_enc, tree)

        print pred

    elif model_type == 4: 
        #read images from scratchs
        print("[INFO] Apply SVM...")
        images, labels = svm.prepare_dataset(params["encodings"])
        pca, clf = svm.train(images, labels, svm_kernal=params["kernel"])
        test_images, test_labels = svm.prepare_dataset(params["image"])

        for i in range(len(test_images)):
            predict_id = svm.predict(test_images[i], pca, clf)
            if predict_id < 0:
                print "image pass"
            elif predict_id == test_labels[i]:
                print "bingle"
            else:
                print "error"
    else: #model_type = 5

        print("[INFO] Apply neural network...")
        model = neural_network.build_model()    
        database = neural_network.train(params["train"], model)
        neural_network.predict(params["image"], database, model)
    #print("[INFO] Apply to old method (the nearest neigher)...")
    #pred = old_method(faces["encodings"], faces["names"], input_enc)


'''
######## show box and picutre
# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
'''
