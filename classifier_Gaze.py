from keras.models import Model, load_model, Sequential
from keras.layers import BatchNormalization, Input, Dense, Dropout, Flatten,LeakyReLU,Concatenate
from keras.callbacks import EarlyStopping
# --------------------------------------------------------------------------------------------------------------------
class classifier_Gaze(object):
    def __init__(self):
        self.name = "FC"
        self.model = []
        self.verbose = 2
        self.epochs = 150
        self.input_image_shape = (64, 64, 3)
        self.input_features_shape = 68*2
        self.init_model()
# ----------------------------------------------------------------------------------------------------------------
    def save_model(self,filename_output):
        self.model.save(filename_output)
        return
# ----------------------------------------------------------------------------------------------------------------
    def load_model(self, filename_weights):
        self.model = load_model(filename_weights)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_model(self):
        layer_features = Input(shape=(self.input_features_shape,))
        #layer_features_001 = BatchNormalization()(layer_features)
        layer_features_002 = Dense(512)(layer_features)
        layer_features_003 = LeakyReLU()(layer_features_002)
        layer_features_004 = Dense(512)(layer_features_003)
        layer_features_005 = LeakyReLU()(layer_features_004)
        layer_features_006 = Dense(512)(layer_features_005)
        layer_features_007 = Dropout(0.5)(layer_features_006)

        layer_image = Input(shape=self.input_image_shape)
        layer_001 = BatchNormalization()(layer_image)
        layer_002 = Flatten()(layer_001)
        layer_003 = Dense(1024)(layer_002)
        layer_004 = LeakyReLU()(layer_003)
        layer_005 = Dense(256)(layer_004)
        layer_006 = LeakyReLU()(layer_005)
        layer_007 = Dense(256)(layer_006)
        layer_008 = LeakyReLU()(layer_007)
        layer_009 = Dense(256)(layer_008)
        layer_009a = Dropout(0.5)(layer_009)

        layer_010 = Concatenate()([layer_features_007,layer_009a])
        layer_011 = Dense(512)(layer_010)
        layer_012 = LeakyReLU()(layer_011)
        layer_013 = Dense(512)(layer_012)
        layer_014 = LeakyReLU()(layer_013)
        layer_015 = Dense(512)(layer_014)
        layer_016 = LeakyReLU()(layer_015)
        layer_017 = Dropout(0.5)(layer_016)
        layer_out = Dense(2)(layer_017)


        self.model = Model(inputs=[layer_image,layer_features],outputs=layer_out)
        #loss = 'mean_absolute_error'
        loss = 'mean_squared_error'
        self.model.compile(optimizer='adam', loss=loss)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def learn(self,X,Y,Z):
        early_stopping_monitor = EarlyStopping(monitor='loss', patience=10)
        self.model.fit(x=[X,Z], y=Y,
                              validation_split=0.3,
                              nb_epoch=self.epochs,
                              verbose=self.verbose,
                              callbacks=[early_stopping_monitor])

# ----------------------------------------------------------------------------------------------------------------------
    def predict(self,X):
        return self.model.predict(X,verbose=0)
# ----------------------------------------------------------------------------------------------------------------------

