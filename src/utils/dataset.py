import cv2
import numpy as np 
import os 
import pandas as pd 
from random import shuffle
from scipy.io import loadmat



class DatasetManager(object):

    def __init__(self, dataset_name='fer2013',
                    dataset_path=None, image_size=(48,48)):
        
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size


        if dataset_name == 'fer2013':
            self.dataset_path = '../dataset/fer2013/fer2013.csv'
    
    def get_data(self):
        return self._load_emotions_dataset()
    

    def _load_emotions_dataset(self):
        try:
            emotion_dataset = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            print("Can't find dataset. Please download it and put under directory ./<project_folder>/dataset/fer2013 ")
            exit(0)
        
        # emotion_dataset = emotion_dataset.loc[emotion_dataset['Usage']==data_split]
        pixels = emotion_dataset['pixels'].tolist()
        faces = []

        for pixel_seq in pixels:
            face = [int(pixel) for pixel in pixel_seq.split(" ")]
            face = np.asarray(face).reshape(48,48)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(emotion_dataset['emotion']).as_matrix()
        return faces, emotions


def get_labels():
    return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    
def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

    

