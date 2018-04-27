# from https://github.com/Lasagne/Lasagne

# noinspection PyPep8Naming
import theano.tensor as T


def conv1d_mc0(layer_in, filters, image_shape=None,
               filter_shape=None, border_mode='valid', subsample=(1,),
               filter_flip=True):
    if image_shape is None:
        image_shape_mc0 = None
    else:
        image_shape_mc0 = (image_shape[0], image_shape[1], 1,
                           image_shape[2])
    if filter_shape is None:
        filter_shape_mc0 = None
    else:
        filter_shape_mc0 = (filter_shape[0], filter_shape[1], 1,
                            filter_shape[2])
    if isinstance(border_mode, tuple):
        (border_mode,) = border_mode
    if isinstance(border_mode, int):
        border_mode = (0, border_mode)
    input_mc0 = layer_in.dimshuffle(0, 1, 'x', 2)
    filters_mc0 = filters.dimshuffle(0, 1, 'x', 2)
    conved = T.nnet.conv2d(
        input_mc0, filters_mc0, image_shape_mc0, filter_shape_mc0,
        subsample=(1, subsample[0]), border_mode=border_mode,
        filter_flip=filter_flip)
    return conved[:, :, 0, :]
