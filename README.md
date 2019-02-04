# Face emotion detection #

<p>The project to detect people's emotions using camera</p>

## Get Started ##

<p>Project requires tensorflow environment, prefered to use 
conda or virtualenv with all installed libraries</p>

- tensorflow 1.9
- matplotlib 3.0.2
- keras 2.2.2
- scikit-image 0.14.0

### Detection ###

- To start object detection run command:
`python main.py detect`

### Train ###

- First need to download dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Move the downloaded file to the datasets directory in the root project.
- Untar the file:
`tar -xzf fer2013.tar`
- Run command to train:
`python main.py train`

