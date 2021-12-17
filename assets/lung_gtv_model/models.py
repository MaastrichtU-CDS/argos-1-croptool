import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, \
    concatenate, BatchNormalization, Conv2DTranspose, PReLU, ReLU


def conv_block(inputs, num_features, kernel_size, params):
    x = Conv2D(num_features, kernel_size, activation=None, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=params.l2_loss))(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization()(x)
    x = Dropout(params.dropout_rate)(x)
    x = Conv2D(num_features, kernel_size, activation=None, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=params.l2_loss))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization()(x)
    return x


def unet(params, num_classes, optimizer, loss):
    input_ct = Input((None, None, params.dict['patch_shape'][-1]),
                     name='CT_input')
    x_1 = conv_block(input_ct, 32, (3, 3), params)
    p_1 = MaxPooling2D((2, 2))(x_1)

    x_2 = conv_block(p_1, 64, (3, 3), params)
    p_2 = MaxPooling2D((2, 2))(x_2)

    x_3 = conv_block(p_2, 128, (3, 3), params)
    p_3 = MaxPooling2D((2, 2))(x_3)

    x_4 = conv_block(p_3, 256, (3, 3), params)
    p_4 = MaxPooling2D((2, 2))(x_4)

    x_5 = conv_block(p_4, 512, (3, 3), params)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x_5)
    u6 = concatenate([u6, x_4])
    x_6 = conv_block(u6, 256, (3, 3), params)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x_6)
    u7 = concatenate([u7, x_3])
    x_7 = conv_block(u7, 128, (3, 3), params)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x_7)
    u8 = concatenate([u8, x_2])
    x_8 = conv_block(u8, 64, (3, 3), params)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x_8)
    u9 = concatenate([u9, x_1])
    x_9 = conv_block(u9, 32, (3, 3), params)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x_9)

    model = Model(inputs=[[input_ct]], outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def attention_block(x, g, out_size):
    w_g = Conv2D(filters=out_size, kernel_size=(1, 1), strides=(1, 1),
                 padding='valid')(g)
    g_1 = BatchNormalization()(w_g)
    
    w_x = Conv2D(filters=out_size, kernel_size=(1, 1), strides=(1, 1),
                 padding='valid')(x)
    x_1 = BatchNormalization()(w_x)  
    
    psi = ReLU()(g_1 + x_1)
    psi = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
                 padding='valid')(psi)
    psi = BatchNormalization()(psi)
    psi = tf.keras.activations.sigmoid(psi)
    return x * psi
 

def attention_unet(params, num_classes, optimizer, loss):
    input_ct = Input((None, None, params.dict['patch_shape'][-1]),
                     name='CT_input')
    # x = BatchNormalization(momentum=0.1)(input_ct)
    x_1 = conv_block(input_ct, 32, (3, 3), params)
    p_1 = MaxPooling2D((2, 2))(x_1)

    x_2 = conv_block(p_1, 64, (3, 3), params)
    p_2 = MaxPooling2D((2, 2))(x_2)

    x_3 = conv_block(p_2, 128, (3, 3), params)
    p_3 = MaxPooling2D((2, 2))(x_3)

    x_4 = conv_block(p_3, 256, (3, 3), params)
    p_4 = MaxPooling2D((2, 2))(x_4)

    x_5 = conv_block(p_4, 512, (3, 3), params)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x_5)
    x_4 = attention_block(x=x_4, g=u6, out_size=128)
    u6 = concatenate([u6, x_4])
    x_6 = conv_block(u6, 256, (3, 3), params)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x_6)
    x_3 = attention_block(x=x_3, g=u7, out_size=64)
    u7 = concatenate([u7, x_3])
    x_7 = conv_block(u7, 128, (3, 3), params)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x_7)
    x_2 = attention_block(x=x_2, g=u8, out_size=32)
    u8 = concatenate([u8, x_2])
    x_8 = conv_block(u8, 64, (3, 3), params)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x_8)
    x_1 = attention_block(x=x_1, g=u9, out_size=16)
    u9 = concatenate([u9, x_1])
    x_9 = conv_block(u9, 32, (3, 3), params)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x_9)

    model = Model(inputs=[[input_ct]], outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def mod_resnet(params, num_classes, optimizer, loss):
    
    def _identity_block(x, num_features, stride=(1, 1), kernel_size=[3, 3]):
        y = Conv2D(num_features, kernel_size, strides=stride, activation=None, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=params.l2_loss))(x)
        y = BatchNormalization()(y)
        y = Conv2D(num_features, kernel_size, strides=stride, activation=None, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=params.l2_loss))(y)
        y = BatchNormalization()(y)
        y = tf.math.add(x, y)
        y = ReLU()(y)
        return y
    
    
    def _convolutional_res_block(x, num_features, stride=(2, 2), kernel_size=[3, 3]):
        y_1 = Conv2D(num_features, kernel_size,strides=stride, activation=None, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=params.l2_loss))(x)
        y_1 = BatchNormalization()(y_1)
        y_2 = Conv2D(num_features, kernel_size, strides=(1, 1), activation=None, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=params.l2_loss))(y_1)
        y_3 = Conv2D(num_features, kernel_size, strides=stride, padding='same')(x)
        y_3 = BatchNormalization()(y_3)
        y = tf.math.add(y_2, y_3)
        y = ReLU()(y)
        return y
    
    def _upsample_block(x, num_features, stride, layer):
        y = x
        dims = tf.shape(x)
        for i in range(0, layer):
            y = tf.image.resize(y, size=[dims[1] * i + 1, dims[2] * i + 1])
            # y = Conv2DTranspose(filters=num_features,
            #                     kernel_size=(2, 2),
            #                     strides=(2, 2),
            #                     padding='same')(y)
        
        y = Conv2D(filters=num_features, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = BatchNormalization()(y)
        return y
    
    input_ct = Input((512, 512, params.dict['patch_shape'][-1]),
                     name='CT_input')
    e_1 = Conv2D(filters=64,
                 kernel_size=(7, 7),
                 strides=(2, 2), padding='same')(input_ct)
    e_1 = BatchNormalization()(e_1)
    e_1 = MaxPooling2D(pool_size=(2, 2))(e_1)
    # print(e_1.shape)
    e_2 = _identity_block(x=e_1,
                          stride=(1, 1),
                          num_features=64)
    e_2 = _identity_block(x=e_2,
                          stride=(1, 1),
                          num_features=64)
    # print(e_2.shape)
    e_3 = _convolutional_res_block(x=e_2,
                                   num_features=128,
                                   stride=(2, 2))
    e_3 = _identity_block(x=e_3,
                          num_features=128,
                          stride=(1, 1))
    # print(e_3.shape)
    e_4 = _convolutional_res_block(x=e_3,
                          num_features=256,
                          stride=(2, 2))
    e_4 = _identity_block(x=e_4,
                          num_features=256,
                          stride=(1, 1))
    # print(e_4.shape)
    e_5 = _convolutional_res_block(x=e_4,
                          num_features=512,
                          stride=(2, 2))
    e_5 = _identity_block(x=e_5,
                          num_features=512,
                          stride=(1, 1))
    # print(e_5.shape)
    up_1 = tf.image.resize(e_3, size=[128, 128])
    up_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(up_1)
    up_1 = BatchNormalization()(up_1)
    # up_1 = _upsample_block(x=e_3,
    #                        num_features=128,
    #                        stride=(2, 2),
    #                        layer=1)
    # print(up_1.shape)
    up_2 = tf.image.resize(e_4, size=[64, 64])
    up_2 = tf.image.resize(up_2, size=[128, 128])
    up_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(up_2)
    up_2 = BatchNormalization()(up_2)
    # up_2 = _upsample_block(x=e_4,
    #                        num_features=128,
    #                        stride=(2, 2),
    #                        layer=2)
    # print(up_2.shape)
    up_3 = tf.image.resize(e_5, size=[32, 32])
    up_3 = tf.image.resize(up_3, size=[64, 64])
    up_3 = tf.image.resize(up_3, size=[128, 128])
    up_3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(up_3)
    up_3 = BatchNormalization()(up_3)
    # up_3 = _upsample_block(x=e_5,
                           # num_features=128,
                           # stride=(2, 2),
                           # layer=3)
    # print(up_3.shape)
    d_1 = tf.concat([up_3, up_2, up_1], axis=-1)
    d_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(d_1)
    d_1 = BatchNormalization()(d_1)
    d_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(d_1)
    d_1 = BatchNormalization()(d_1)
    # print('d_1')
    # print(d_1.shape)
    d_2 = tf.math.add(e_2, d_1)
    d_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(d_2)
    d_2 = BatchNormalization()(d_2)
    # print('d_2')
    # print(d_2.shape)
    
    d_3 = Conv2DTranspose(filters=64,
                          kernel_size=(3, 3),
                          strides=(2, 2),
                          padding='same')(d_2)
    d_3 = BatchNormalization()(d_3)
    d_3 = Conv2DTranspose(filters=64,
                          kernel_size=(3, 3),
                          strides=(2, 2),
                          padding='same')(d_3)
    d_3 = BatchNormalization()(d_3)
    # print()
    # print(d_3.shape)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(d_3)

    model = Model(inputs=[[input_ct]], outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
