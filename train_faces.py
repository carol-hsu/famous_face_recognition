# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
from famous_face_recognition import knn, decision_tree

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
    ap.add_argument("-e", "--encodings", required=True,
            help="path to serialized db of facial encodings")
    ap.add_argument("-i", "--image", required=True,
            help="path to input image")
    ap.add_argument("-m", "--learning-model", type=int, default=0,
            help="Use which model for learning: 0=none, 1=decision tree, 2=neural network, 3=boost, 4=svm, 5=knn")
    ap.add_argument("-n", "--number-neighbors", type=int, default=8,
            help="neighbor number for knn model (default: 8)")
    params = vars(ap.parse_args())

    #load encodings
    faces = load_encodings(params["encodings"])
    
    #trained + inference + test    
    model_type = params["learning_model"]

    input_enc, input_loc = encode_image(params["image"])

    pred = ("unknown", -1)

    if model_type == 1 or model_type == 3:
        print("[INFO] Apply decision tree method...")
        tree = decision_tree.train(faces["encodings"], faces["names"])
        decision_tree.predict(input_enc, tree)
        ## decision_tree.main((faces["encodings"], faces["names"])
    elif model_type == 2:
        print("[INFO] Apply neural network...")
    elif model_type == 4:
        print("[INFO] Apply SVM...")
    elif model_type == 5:
        print("[INFO] Apply knn model with k = "+str(params["number_neighbors"]))
        knn_clfier = knn.train(faces["encodings"], faces["names"], n_neighbors=params["number_neighbors"])
        pred = knn.predict(input_enc, knn_clf=knn_clfier)
    else:
        print("[INFO] Apply to old method (the nearest neigher)...")
        pred = old_method(faces["encodings"], faces["names"], input_enc)

    print pred
    #encoding the input file
    #show result
    #show picture

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
