from statistics import mode 
import os 
import sys
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np 

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from src.utils.dataset import DatasetManager, split_data, get_labels
from src.utils.visualizer import pretty_imshow, make_mosaic
from src.utils.preprocessing import preprocess_input
from src.model.net import neural_net
from src.model.config import Config

from src.utils.detection import detect_faces
from src.utils.detection import draw_text
from src.utils.detection import draw_bounding_box
from src.utils.detection import apply_offsets
from src.utils.detection import load_detection_model



def displayData():
    loader = DatasetManager()
    faces, emotions = loader.get_data()
    pretty_imshow(plt.gca(), make_mosaic(faces[:4], 2, 2), cmap='gray')
    plt.show()


def train(config):

    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)


    model = neural_net(config.input_shape, config.num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])
    
    model.summary()

    dataset_name = 'fer2013'

    print("Train on {}".format(dataset_name))

    log_files = config.base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_files, append=False)
    early_stop = EarlyStopping('val_loss', patience=config.patience)
    reduce_lr = ReduceLROnPlateau('var_loss', factor=0.1, verbose=1, patience = int(config.patience/4))
    
    trained_models_path = config.base_path + dataset_name + '_neural_model'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)

    callbacks = [model_checkpoint, reduce_lr, early_stop, csv_logger]

    data_loader = DatasetManager(dataset_name, image_size=config.input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(faces, emotions, config.validation_split)
    train_faces, train_emotions = train_data 
    model.fit_generator(data_generator.flow(train_faces, train_emotions, config.batch_size),
                            steps_per_epoch=len(train_faces) / config.batch_size,
                            epochs=config.num_epochs, verbose=1, callbacks=callbacks,
                            validation_data=val_data)
def detect(config):

    cascade_model_path = os.path.join(ROOT_DIR,'src/model/haarcascade_frontalface_default.xml')
    emotion_model_path = os.path.join(ROOT_DIR,'trained_models/fer2013_neural_model.184-0.63.hdf5')
    emotion_labels = get_labels()


    face_detection = load_detection_model(cascade_model_path)
    emotion_classifier = load_model(emotion_model_path)


    emotion_target_size = emotion_classifier.input_shape[1:3]

    emotion_window = []

    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)

    while True:

        bgr_image = video_capture.read()[1]
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, config.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)
            
            if len(emotion_window) > config.frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                    color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Emotion face model"
    )

    parser.add_argument('command', metavar='<command>', help="''")

    args = parser.parse_args()

    if args.command == 'display_data':
        displayData()
    
    if args.command == 'train':
        config = Config()
        config.display()

        train(config)
    
    if args.command == 'detect':
        config = Config()
        config.display()

        detect(config)

        