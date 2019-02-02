
class Config(object):

    batch_size = 32

    num_epochs = 10000

    input_shape = (64, 64, 1)

    validation_split = .2

    verbose = 1

    num_classes = 7

    patience = 50

    frame_window = 10

    emotion_offsets = (20, 40)

    base_path = '../trained_models/'

    def display(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")