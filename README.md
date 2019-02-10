# [Famous face recognition](https://github.com/carol-hsu/famous_face_recognition)

Studying supervised learnings by building models from scatch.

The implementation of models is referenced by several others projects:
- [Face recognition program](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
- [Image searching by Microsoft API(Bing)](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)
- [Python face recognition package](https://github.com/ageitgey/face_recognition)
- [Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [K-NN](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)
- [SVM](https://github.com/ahhda/Face-Recogntion)
- [Neural network](https://github.com/amenglong/face_recognition_cnn)

This is for self study/practice/school assignment :octocat: no profit for purpose!

Following instructions show you how to play with this repo:

## Environment setup
Make sure you have Python on your machine.
After you pull this repo, following commands helps you installing the required packages.

```
// virtual environment is recommended, but optional
$ virtualenv venv

(venv) $ pip install -r requirements.txt
```
## Encoding images
Part of learning algorithms rely on encoding datas, instead of raw images. Let's encode the images first!
Before encoding, put your face images in nested directories: a secondary layer of directories which separate your images by **names**.
for example:

```
$ tree pk_dataset_test
pk_dataset_test/
├── keira_knightley
│   ├── 001.jpg
│   └── 002.jpg
└── natalie_portman
    ├── 001.jpg
    └── 002.jpg
```

no matter how many layers in upper directory, be aware to separate the files 

Next, encoding your images with flag `-i` for path of image directory, and `-e` the name of out encodings
```
$ python encode_faces.py -i $ROOT_DIRECTORY_OF_IMAGES -e "OUTPUT_NAME"
```
i.e. 
```
$ python encode_faces.py -i ./pk_dataset_test -e pk_test.pickle
```

## Run algorithms

This program covers five algorithms. Python file `face_recongnition.py` is the entry point.

- Check help messages `-h` if you forget any flag usage.
```
$ python face_recongnition.py -h
Using TensorFlow backend.
usage: face_recongnition.py [-h] [-t TRAIN] [-i TEST] [-v VALIDATE]
                            [-m LEARNING_MODEL] [-n NUMBER_NEIGHBORS]
                            [-k KERNEL] [-b BOOSTING_ESTIMATORS]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        directory of training set/encodings of training set
  -i TEST, --test TEST  directory of testing set/encodings of testing set
  -v VALIDATE, --validate VALIDATE
                        directory of validating set/encoding of validating set
  -m LEARNING_MODEL, --learning-model LEARNING_MODEL
                        Use which model for learning: 1=knn, 2=decision tree,
                        3=boosting, 4=svm, 5=neural network
  -n NUMBER_NEIGHBORS, --number-neighbors NUMBER_NEIGHBORS
                        neighbor number for knn model (default: 8)
  -k KERNEL, --kernel KERNEL
                        kernel for svm model,
                        linear/poly/rbf/sigmoid/precomputed (default: rbf)
  -b BOOSTING_ESTIMATORS, --boosting-estimators BOOSTING_ESTIMATORS
                        the number of estimators for boosting
```
These flags are not required all the time, but please use them respectively based on the algorithm you want to try.




