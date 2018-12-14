from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Lambda, Input
from keras import backend as K
import numpy as np
import cv2
from keras_vggface.utils import preprocess_input
from keras.utils import plot_model
from sklearn.preprocessing import normalize
import config


def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    '''
    This function is inspired by the FaceNet paper https://arxiv.org/abs/1503.03832
    this loss function is designed to keep distance of items within a class short
    while keeping the distance of items between classes far.
    :param inputs: anchor and positive are items from the same class while negative
    item is from another class.
    :param margin: 'maxplus', 'softplus' this is to avoid negative loss
    :return: loss tha is the difference between negative and positive distances
    '''
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance

    if margin == 'maxplus': # Using maxplus will give us the benefit of accounting all distances inside a batch
        # as well as defining the difference (margin) between negative gap and positive gap
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)


def triplet_loss_np(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = np.log(1 + np.exp(loss))
    return loss.mean()


def check_loss():
    batch_size = 10
    shape = (batch_size, 4096)

    p1 = normalize(np.random.random(shape))
    n = normalize(np.random.random(shape))
    p2 = normalize(np.random.random(shape))

    input_tensor = [K.variable(p1), K.variable(n), K.variable(p2)]
    out1 = K.eval(triplet_loss(input_tensor))
    input_np = [p1, n, p2]
    out2 = triplet_loss_np(input_np)

    assert out1.shape == out2.shape
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1 - out2))


def make_models():

    '''
    Creating two models one is the imported resnet50 with a normaliser layer to project all feature vectors
    onto a hypersphere. We use this for creating the second model with triplet outputs and inputs as well as
    controlling the trainable layers.
    :return: the actual resnet model with a normaliser on top and the triplet loss model for further training.
    '''

    # Import resnet as a feature extractor i.e. no prediction layer
    vggface = VGGFace(model='resnet50', include_top=False)
    # Add l2 normalisation to map all feature vectors on a hypersphere.
    norm = Lambda(lambda x: K.l2_normalize(x, axis=3))(vggface.layers[-1].output)
    # Make the new model
    ResnetModel = Model(vggface.layers[0].input, norm)
    input_shape = (config.image_size, config.image_size, config.channels)
    # Create empty tensors (placeholders) for each input.
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    anchor_feature = ResnetModel(anchor_input)
    positive_feature = ResnetModel(positive_input)
    negative_feature = ResnetModel(negative_input)
    triplet_input = [anchor_input, positive_input, negative_input]
    triplet_output = [anchor_feature, positive_feature, negative_feature]
    # Create the model
    TripletModel = Model(triplet_input, triplet_output)
    TripletModel.add_loss(triplet_loss(triplet_output))
    return ResnetModel, TripletModel


# if __name__=='__main__':
#     check_loss()
#     breakpoint()
#     vggface = VGGFace(model='resnet50', include_top=False)
#     # Add l2 normalisation to map all feature vectors on a hypersphere.
#     norm = Lambda(lambda x: K.l2_normalize(x, axis=0))(vggface.layers[-1].output)
#     # Make the new model
#     ResnetModel = Model(vggface.layers[0].input, norm)
#     im = cv2.imread('/home/vms/Desktop/training/dataset50/1,m,40/2 - 1.jpg')
#     # im = cv2.resize(im, (config.image_size, config.image_size))
#     # im = np.expand_dims(im, axis=0)
#     # im = preprocess_input(im.astype(np.float64), version=2)
#     # k = ResnetModel.predict(im)
#
#
#
#     ResnetModel.summary()
#     # print the shape of features
#     print(ResnetModel.layers[-1].output.shape)