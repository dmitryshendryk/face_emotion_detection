import os 
import sys
import matplotlib.pyplot as plt
import argparse

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from src.utils.dataset import DatasetManager, split_data
from src.utils.visualizer import pretty_imshow, make_mosaic
from src.model.net import neural_net
from src.utils.preprocessing import preprocess_input
from src.model.config import Config




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

        