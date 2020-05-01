import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.ndimage.morphology import distance_transform_edt
import scipy.ndimage
import os
from skimage.io import imsave
import numpy as np
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from src.Models.SegmentationNetworks import unet_3D_ResNet_DB
from src.Utils.LossFunctions import weighted_dice_loss3D, dice_coef
from src.Utils.DataPreprocessing_Segmentation import preprocessing_train, preprocessing_valid
from src.Utils.DataLoaders import createdatatensor, datagenerator, createdatatensorwithouthe

class Rectum_segmentation():

    def __init__(self):
        print("-------------------------------------------------------------------------------------------------------")
        print("INITIALIZING VARIABLES")
        print("-------------------------------------------------------------------------------------------------------")
        self.img_rows = 160
        self.img_cols = 160
        self.img_slcs = 64
        self.batch_size = 4
        self.img_shape = (self.img_rows, self.img_cols, self.img_slcs, 1)
        self.starting_filter = 16
        self.poolsize = (2, 2, 1)
        self.number_of_pool = 5
        self.epochs = 100
        self.main_pred_dir = 'data/M1_Rectum'
        self.roi_interest = 4

        pat = np.load('data/Patlist/train_pat_postop.npy')
        sub = np.load('data/Patlist/train_sub_pat_postop.npy')
        z = np.load('data/Patlist/z_train_postop.npy')
        split = int(.9*len(pat))
        end = len(pat)
        self.train_pat = pat[0:split]
        self.train_pat_sublist = sub[0:split]
        self.z_train = z[0:split]
        self.valid_pat = pat[split:end]
        self.valid_sub_pat = sub[split:end]
        self.z_valid = z[split:end]

    def train(self):
        # preporocess train and valid data, calculate slicing volume - x,y,z
        self.z_starttrain, self.z_endtrain, self.x_centtrain, self.y_centtrain = preprocessing_train(
            self.main_pred_dir, self.train_pat, self.train_pat_sublist, self.z_train, self.roi_interest, self.img_slcs, justload=1)
        self.z_startvalid, self.z_endvalid, self.x_centvalid, self.y_centvalid = preprocessing_valid(
            self.main_pred_dir, self.valid_pat, self.valid_sub_pat, self.z_valid, self.roi_interest, self.img_slcs, justload=1)

        ## if creating and loading entire tensor
        x_train, y_train = createdatatensor(self.main_pred_dir, self.img_rows, self.img_cols, self.img_slcs,self.train_pat, self.train_pat_sublist, self.z_starttrain,
                                            self.z_endtrain, self.x_centtrain, self.y_centtrain, self.roi_interest,
                                             't', justload = 1)
        x_valid, y_valid = createdatatensor(self.main_pred_dir, self.img_rows, self.img_cols, self.img_slcs,self.valid_pat, self.valid_sub_pat, self.z_startvalid,
                                            self.z_endvalid, self.x_centvalid, self.y_centvalid, self.roi_interest,
                                            'v', justload = 1)

        ##if just loading
        # x_train, y_train = createdatatensor(self.main_pred_dir, trainorvalid='t', justload=1)
        # x_valid, y_valid = createdatatensor(self.main_pred_dir, trainorvalid='v', justload=1)

        ## if using Datagenerators
        # self.traindatagenerator() = datagenerator(self.main_pred_dir,self.batch_size, self.img_rows, self.img_cols, self.img_slcs, self.roi_interest, 't')
        # self.validdatagenerator() = datagenerator(self.main_pred_dir, self.batch_size, self.img_rows, self.img_cols,self.img_slcs, self.roi_interest, 'v')
        self.Rectum_network = unet_3D_ResNet_DB(self.img_rows, self.img_cols,self.img_slcs, channels_in=1, channels_out=1,
            starting_filter_number=self.starting_filter,kernelsize=(3, 3, 3), number_of_pool=self.number_of_pool, poolsize=self.poolsize, filter_rate=2,
            final_activation='sigmoid')
        self.Rectum_network.compile(optimizer=Adam(lr=1e-2), loss=weighted_dice_loss3D, metrics=[dice_coef])
        self.Rectum_network.load_weights('Rectum_weights.h5')
        model_checkpoint = ModelCheckpoint('Rectum_weights.h5', monitor='val_dice_coef', save_best_only=True, save_weights_only = True, mode='max')
        self.Rectum_network.fit(x_train,y_train, batch_size = self.batch_size, epochs= self.epochs, verbose=1 , callbacks=[model_checkpoint],
                                 validation_data = [x_valid,y_valid])
        # if using Datagenerators
        # self.FM2_network.fit(self.traindatagenerator(), steps_per_epoch=int(len(self.train_pat)/self.batch_size), epochs= self.epochs, verbose=1,
        #                                      validation_data = self.validdatagenerator(), validation_steps = int(len(self.valid_pat)/self.batch_size), callbacks=[model_checkpoint])


if __name__ == '__main__':
    Rectum_model = Rectum_segmentation()
    Rectum_model.train()
















