from keras import backend as K
from keras.layers import Lambda, Multiply


def Spatial_multiply(x):
    image, mask = x
    mask = Lambda(lambda x: K.squeeze(x, axis=-1))(mask)
    mask = Lambda(lambda x: K.stack([x] * image.shape.dims[-1].value, axis=-1))(mask)
    mask_image = Multiply()([mask, image])

    return mask_image


def Channel_multiply(x):
    image, mask_weight = x
    mask = Lambda(lambda x: K.stack([x] * image.shape.dims[1].value, axis=-1))(mask_weight)
    mask = Lambda(lambda x: K.stack([x] * image.shape.dims[2].value, axis=-1))(mask)
    mask = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 3, 1]))(mask)
    mask_image = Multiply()([mask, image])

    return mask_image
