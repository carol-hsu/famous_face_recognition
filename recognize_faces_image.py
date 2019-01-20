# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2

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
            print("[INFO] Finish loading! Get "+str(len(data["names"]))+" ones ")
            break

    return data


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
    	help="path to serialized db of facial encodings")
#    ap.add_argument("-i", "--image", required=True,
#    	help="path to input image")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    	help="face detection model to use: either `hog` or `cnn`")
    ap.add_argument("-m", "--learning-model", type=int, default=0,
            help="Use which model for learning: 0=none, 1=decision tree, 2=neural network, 3=boost, 4=svm, 5=knn")
    params = vars(ap.parse_args())

    #load encodings
    encodings = load_encodings(params["encodings"])
    #trained + inference + test    
    #encoding the input file
    #show result
    #show picture

'''
# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []


## update encodings with the algos
# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(data["encodings"], encoding)
        confidence = face_recognition.face_distance(data["encodings"], encoding)
	name = "Unknown"

	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}
                confid_value = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = data["names"][i]
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
	names.append(name)
        print name
        print 1-min(confid_value[name])

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
