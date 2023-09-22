import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2

class EMA_Unet:
    def __init__(self, momentum=0.999, batchnorm=False):
        self.student = EMA_Unet.build_unet_model(batchnorm=batchnorm)
        self.teacher = EMA_Unet.build_unet_model(batchnorm=batchnorm)

        self.momentum = momentum
        # Initially, params_t = params_s
        self.teacher.set_weights(self.student.get_weights())
    
    def update_ema_params(self, step, ema=False):
        alpha = min(self.momentum, 1 - (1/(1 + step)))
        if(step == 1 or not ema):
            self.teacher.set_weights(self.student.get_weights())
        else:
            for i, s_layer in enumerate(self.student.layers):
                s_params = s_layer.weights
                
                if(len(s_params) > 0):
                    updated_params = []
                    for j in range(len(s_params)):
                        t_params = self.teacher.layers[i].weights
                        updated_params.append(alpha * t_params[j] + (1 - alpha) * s_params[j])

                    self.teacher.layers[i].set_weights(updated_params)

    @staticmethod
    def sigmoid_rampup(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


    @staticmethod
    def build_unet_model(batchnorm=False):
        inputs = Input(shape=(256, 256, 3))
        init = 'he_uniform' # l1(2e-4)
        reg  = l1(2e-04)

        ## Encoding path ##
        conv_1_a = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(inputs)
        conv_1_b = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_1_a)
        if(batchnorm): conv_1_b = BatchNormalization()(conv_1_b)
        conv_1_b = Dropout(0.5)(conv_1_b)

        pool = MaxPooling2D(pool_size=(2,2))(conv_1_b)

        conv_2_a = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_2_b = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_2_a)
        if(batchnorm): conv_2_b = BatchNormalization()(conv_2_b)
        conv_2_b = Dropout(0.5)(conv_2_b)

        pool = MaxPooling2D(pool_size=(2,2))(conv_2_b)

        conv_3_a = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_3_b = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_3_a)
        if(batchnorm): conv_3_b = BatchNormalization()(conv_3_b)
        conv_3_b = Dropout(0.5)(conv_3_b)

        pool = MaxPooling2D(pool_size=(2,2))(conv_3_b)

        conv_4_a = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_4_b = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_4_a)
        if(batchnorm): conv_4_b = BatchNormalization()(conv_4_b)
        conv_4_b = Dropout(0.5)(conv_4_b)

        pool = MaxPooling2D(pool_size=(2,2))(conv_4_b)

        conv_5_a = Conv2D(1024, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_5_b = Conv2D(1024, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_5_a)
        if(batchnorm): conv_5_b = BatchNormalization()(conv_5_b)
        conv_5_b = Dropout(0.5)(conv_5_b)

        ## Decoding path ##
        up1 = UpSampling2D(size=(2,2))(conv_5_b)
        up1 = Concatenate(axis=3)([conv_4_b, up1])

        upconv_1_a = Conv2D(512, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up1)
        upconv_1_b = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_1_a)
        if(batchnorm): upconv_1_b = BatchNormalization()(upconv_1_b)
        upconv_1_b = Dropout(0.5)(upconv_1_b)

        up2 = UpSampling2D(size=(2,2))(upconv_1_b)
        up2 = Concatenate(axis=3)([conv_3_b, up2])

        upconv_2_a = Conv2D(256, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up2)
        upconv_2_b = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_2_a)
        if(batchnorm): upconv_2_b = BatchNormalization()(upconv_2_b)
        upconv_2_b = Dropout(0.5)(upconv_2_b)

        up3 = UpSampling2D(size=(2,2))(upconv_2_b)
        up3 = Concatenate(axis=3)([conv_2_b, up3])

        upconv_3_a = Conv2D(128, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up3)
        upconv_3_b = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_3_a)
        if(batchnorm): upconv_3_b = BatchNormalization()(upconv_3_b)
        upconv_3_b = Dropout(0.5)(upconv_3_b)

        up4 = UpSampling2D(size=(2,2))(upconv_3_b)
        up4 = Concatenate(axis=3)([conv_1_b, up4])

        upconv_4_a = Conv2D(64, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up4)
        upconv_4_b = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_4_a)
        if(batchnorm): upconv_4_b = BatchNormalization()(upconv_4_b)
        upconv_4_b = Dropout(0.5)(upconv_4_b)
        logits = Conv2D(1, kernel_size=3, activation=None, padding='same', kernel_initializer=init, kernel_regularizer=reg)(upconv_4_b)

        output = Activation('sigmoid')(logits)

        model = Model(inputs=inputs, outputs=[output, logits])
        return model

model = build_unet_model()
# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)