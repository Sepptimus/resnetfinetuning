from ResnetFace import make_models
from keras.optimizers import Adam
from data import DataReader, TripletGenerator
import config
from os.path import join
'''
This fine-tuning is inspired by https://github.com/Ao-Lee/Vgg-Face-Fine-tune as well as 
the triplet loss function appeared in the FaceNet paper https://arxiv.org/abs/1503.03832 
I replaced the VGG model with ResNet50 trained on VGGFace2.
'''

if __name__ == '__main__':

    ResnetModel, TripletModel = make_models()
    # train_data = DataReader(dir_images=config.path_LFW)
    # train_data = LFWReader(dir_images=config.path_LFW)
    train_data = DataReader(dir_images=config.train_data)
    train_generator = TripletGenerator(train_data)
    test_data = DataReader(dir_images=config.test_data)
    test_generator = TripletGenerator(test_data)

    # Set the numbers of trainable layers
    for layer in ResnetModel.layers[-10:]:
        print(layer.name)
        layer.trainable = True
    for layer in ResnetModel.layers[:-10]:
        print(layer.name)
        layer.trainable = False

    # for layer in ResnetModel.layers:
    #     layer.trainable = True

    TripletModel.compile(loss=None, optimizer=Adam(2e-5))
    history = TripletModel.fit_generator(train_generator,
                                          validation_data=train_generator,
                                          epochs=5,
                                          verbose=1,
                                          workers=4,
                                          steps_per_epoch=20,
                                          validation_steps=10,
                                          use_multiprocessing=True)

    # ResnetModel.save_weights(join(config.model_path, 'NewModel.h5'))
