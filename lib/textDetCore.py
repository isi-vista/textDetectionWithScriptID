'''
This file defines the required deep nerual network layers and models for text detection.

NOTE: 
1. Unless you know what you are doing, 
   or you don't care the use of pretrained models, 
   please don't make any modification to this file.
2. This file is compatible with Python2 and Python3
'''
from keras.layers import merge, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

def bn_conv2d( x, nfilters, kernal_size, name ) :
    '''Basic module of "Conv2D + Batch Normalization + Relu"
    '''
    x = Conv2D( nfilters, kernal_size, padding = 'same', activation = None, name = name + '-c' )(x)
    x = BatchNormalization( name = name + '-bn' )(x)
    x = Activation('relu', name = name + '-re' )(x)
    return x

def google_inception(x, nb_inc=16, inc_filt_list=[(1,1), (3,3), (5,5)], name='uinc') :
    '''Basic Google Inception Module
    '''
    uc_list = []
    for idx, ftuple in enumerate( inc_filt_list ) :
        uc = Conv2D( nb_inc, ftuple, padding='same', name=name+'_c%d' % idx)(x)
        uc_list.append(uc)
    if ( len( uc_list ) > 1 ) :
        uc_merge = Concatenate( axis=-1, name=name+'_merge')(uc_list)
    else :
        uc_merge = uc_list[0]
    uc_norm = BatchNormalization(name=name+'_bn')(uc_merge)
    uc_relu = Activation('relu', name=name+'_re')(uc_norm)
    return uc_relu

class BilinearUpSampling( Layer ) :
    '''Keras Custom Layer for Bilinear Upsampling
    '''
    def __init__( self, height_factor = 2, width_factor = 2, **kwargs ) :
        self.height_factor = height_factor
        self.width_factor = width_factor
        super( BilinearUpSampling, self ).__init__( **kwargs )
    def build( self, input_shape ) :
        super( BilinearUpSampling, self ).build( input_shape )
    def call( self, x ):
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array([self.height_factor, self.width_factor]).astype('int32'))
        x = tf.image.resize_bilinear(x, new_shape, align_corners=True)
        x.set_shape((None, original_shape[1] * self.height_factor if original_shape[1] is not None else None,
                     original_shape[2] * self.width_factor if original_shape[2] is not None else None, None))
        return x
    def compute_output_shape( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_filts = input_shape
        out_nb_rows = None if ( nb_rows is None ) else nb_rows * self.height_factor
        out_nb_cols = None if ( nb_cols is None ) else nb_cols * self.width_factor
        return tuple( [ bsize, out_nb_rows, out_nb_cols, nb_filts ] )

def create_scriptID_model( input_shape = (None, None, 3), base = 16, name = 'scriptID') :
    ''' create a Fully Convolutional Network model for script ID prediction
    '''
    ii = Input( shape = input_shape, name = 'image_in' )
    # block #1
    bname = name + 'b1'
    x = bn_conv2d( ii, base, (7,7), name = bname+'c1' )
    x = bn_conv2d( x, base, (1,1), name = bname+'c2' )
    x = bn_conv2d( x, base, (5,5), name = bname+'c3' )
    x = bn_conv2d( x, base, (1,1), name = bname+'c4' )
    x = MaxPooling2D((3,3), strides = (2,2), padding = 'same', name = bname+'mp' )(x)
    # block #2, 3, 4
    for blk in range(2,5) :
        base *= 2
        bname = name + 'b%d' % blk
        x = bn_conv2d( x, base, (3,3), name = bname+'c1' )
        x = bn_conv2d( x, base, (1,1), name = bname+'c2' )
        x = bn_conv2d( x, base, (3,3), name = bname+'c3' )
        x = bn_conv2d( x, base, (1,1), name = bname+'c4' )
        x = MaxPooling2D((3,3), strides = (2,2),  padding = 'same', name = bname+'mp' )(x)
    # language histogram
    lx = bn_conv2d(  x, base * 4, (3,3), name = name+'l2-c1' )
    lx = bn_conv2d( lx, base * 4, (1,1), name = name+'l2-c2' )
    lx = bn_conv2d( lx, base * 4, (3,3), name = name+'l2-c3' )
    lx = bn_conv2d( lx, base * 4, (1,1), name = name+'l2-c4' )
    lx = MaxPooling2D((3,3), strides = (2,2),  padding = 'same', name = name+'l2-mp' )(lx)
    lx = GlobalAveragePooling2D( name = name+'l-pool')(lx)
    lx = Dropout(0.5, name = name+'l-d1')(lx)
    lx = Dense(128, name = name+'l-fc1', activation = 'relu')(lx)
    lx = Dropout(0.5, name = name+'l-d2')(lx)
    ly = Dense(7, name = name+'pred', activation = 'softmax' )(lx)
    return Model( inputs = ii, outputs =ly, name=name )

def create_textDet_model( input_shape = ( None, None, 3 ), base = 16, name = 'textDet' ) :
    ii = Input( shape = input_shape, name = 'image_in' )
    # block #1
    bname = name+'b1'
    x = bn_conv2d( ii, base, (7,7), name = bname+'c1' )
    x = bn_conv2d( x, base, (1,1), name = bname+'c2' )
    x = bn_conv2d( x, base, (5,5), name = bname+'c3' )
    x = bn_conv2d( x, base, (1,1), name = bname+'c4' )
    x = MaxPooling2D((3,3), strides = (2,2), padding = 'same', name = bname+'mp' )(x)
    # block #2, 3, 4
    for blk in range(2,5) :
        base *= 2
        bname = name+'b%d' % blk
        x = bn_conv2d( x, base, (3,3), name = bname+'c1' )
        x = bn_conv2d( x, base, (1,1), name = bname+'c2' )
        x = bn_conv2d( x, base, (3,3), name = bname+'c3' )
        x = bn_conv2d( x, base, (1,1), name = bname+'c4' )
        x = MaxPooling2D((3,3), strides = (2,2),  padding = 'same', name = bname+'mp' )(x)
    # deconvolution layer
    x = google_inception( x, 32, [1,3], name = name+'d1' )
    x = BilinearUpSampling( name= name+'d1x2')(x)
    x = google_inception( x, 24, [3,5], name = name+'d2' )
    x = BilinearUpSampling( name= name+'d2x2')(x)
    x = google_inception( x, 16, [5,7], name = name+'d3' )
    x = BilinearUpSampling( name= name+'d3x2')(x)
    x = google_inception( x, 8, [3,5,7], name = name+'d4' )
    x = BilinearUpSampling( name= name+'d4x2')(x)
    x = google_inception( x, 4, [3,5,7,9], name = name+'d5' )
    y = Conv2D( 3,(11,11), padding = 'same', name =  name+'pred', activation = 'softmax' )(x)
    return Model( inputs = ii, outputs = y, name = name )
