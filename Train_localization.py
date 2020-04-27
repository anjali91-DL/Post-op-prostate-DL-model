import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
tf.compat.v1.disable_eager_execution()
from src.Models.LocalizationNetwork import unet_2D
from src.Utils.LossFunctions import channel_loss, channel_Dice, channel_RSloss, channel_RSDice
from src.Utils.Data_Utils import save_coarse_predictions
from numpy import load

class postop_localization():

    def __init__(self):
        print("-------------------------------------------------------------------------------------------------------")
        print("INITIALIZING VARIABLES")
        print("-------------------------------------------------------------------------------------------------------")
        self.data_dir = '/data/mnt/share/dan/edrive/Anjali_Backup/BED_OAR_MASKS'
        self.loc_dir = '/data/s426200/CTVSeg'
        self.pat = np.load('/data/s426200/CTVSeg/ALL_train_pats.npy')
        split = int(.9*len(self.pat))
        end = len(self.pat)
        self.pat = np.load('/data/s426200/CTVSeg/ALL_train_pats.npy')
        self.Train_pat = self.pat[0:split]
        print('train total', len(self.Train_pat))
        self.Train_sub_pat = np.load('/data/s426200/CTVSeg/ALL_train_pat_sub.npy')[0:split]
        self.VALID_pat = self.pat[split:end]
        self.VALID_sub_pat = np.load('/data/s426200/CTVSeg/ALL_train_pat_sub.npy')[split:end]
        print('valid total',len(self.VALID_pat))

    def load_real_samples(self, filename):
        # load the dataset
        data = load(filename)
        # unpack arrays
        X1, X2, X3, X4, X5, X6, X7 = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data['arr_6']

        X1 = np.tanh(X1[:,100:420,100:420, :] / 1000)
        Xroi = np.ndarray((X1.shape[0], X1.shape[1], X1.shape[2], 4), dtype=np.float32)
        Xroi[:, :, :, 0:1] = X2[:,100:420,100:420, :] + X3[:,100:420,100:420, :]
        Xroi[:, :, :, 1:2] = X4[:,100:420,100:420, :]
        Xroi[:, :, :, 2:3] = X5[:,100:420,100:420, :]
        Xroi[:, :, :, 3:4] = X6[:,100:420,100:420, :]
        return [X1, Xroi]

    def train(self):
        TrainCT, TrainROI = self.load_real_samples('/data/s426200/LocalizationpostopCT_train_local.npz')
        ValidCT, ValidROI = self.load_real_samples('/data/s426200/LocalizationpostopCT_valid_local.npz')
        print("RUNNING MODEL")

        self.loc_network = unet_2D(320, 320, channels_in=1, channels_out=5,
            starting_filter_number=32, kernelsize=(3, 3), number_of_pool=4, poolsize=(2,2), filter_rate=2,
            dropout_rate=0.3,final_activation='sigmoid')
        self.loc_network.compile(optimizer=Adam(lr=1e-3), loss=channel_loss, metrics=[channel_Dice])
        model_checkpoint = ModelCheckpoint('localizationweights_1.h5', monitor='val_channel_RSDice',
                                           save_best_only=True, mode='max')
        self.loc_network.fit(TrainCT, TrainROI, batch_size=32, epochs=30, verbose=1, shuffle=True,
                                             validation_data = [ValidCT, ValidROI],
                            callbacks=[model_checkpoint])
        self.loc_network.save('locnetwork_1.h5')
        cust = {'channel_Dice': channel_Dice, 'channel_loss': channel_loss}
        self.loc_network = load_model('locnetwork_1.h5', cust)
        self.loc_network.compile(optimizer=Adam(lr=1e-3), loss=channel_RSloss, metrics=[channel_RSDice])
        model_checkpoint = ModelCheckpoint('localizationweights_2.h5', monitor='val_channel_RSDice', save_best_only=True, mode='max')
        self.loc_network.fit(TrainCT, TrainROI, batch_size=32, epochs=10, verbose=1, shuffle=True,
                                             validation_data = [ValidCT, ValidROI],
                            callbacks=[model_checkpoint])
        self.loc_network.save('locnetwork_2.h5')
        Valid_pred = self.loc_network.predict(ValidCT, batch_size=32, verbose=1)
        save_coarse_predictions(self.loc_dir, Valid_pred)

if __name__ == '__main__':
    loc_model = postop_localization()
    loc_model.train()