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
from src.Utils.Data_Utils import z_calc,calculating_slicesofinterest_segtrain, \
    calculating_slicesofinterestfor3D, calculate_centroid_segtrain, calculate_centroid_segvalid, image_histogram_equalization

class Rectum_segmentation():

    def __init__(self):
        print("-------------------------------------------------------------------------------------------------------")
        print("INITIALIZING VARIABLES")
        print("-------------------------------------------------------------------------------------------------------")
        self.img_rows = 160
        self.img_cols = 160
        self.img_slcs = 64
        self.channels_in = 1
        self.channels_out = 1
        self.batch_size = 4
        self.img_shape = (self.img_rows, self.img_cols, self.img_slcs, self.channels_in)
        self.starting_filter = 16
        self.poolsize = (2, 2, 1)
        self.number_of_pool = 5
        self.epochs = 200
        self.pat = np.load('/data/ALL_train_pats.npy')
        split = int(.9*len(self.pat))
        end = len(self.pat)
        self.train_pat = self.pat[0:split]
        print('train total', len(self.train_pat))
        self.train_sub_pat = np.load('/data/ALL_train_pat_sub.npy')[0:split]
        self.valid_pat = self.pat[split:end]
        self.valid_sub_pat = np.load('/data/ALL_train_pat_sub.npy')[split:end]
        print('valid total',len(self.valid_pat))
        self.data_dir = '/data'
        self.main_pred_dir = '/data/M1_Rectum'
        self.roi_interest = 4

    def preprocessing_train(self):
         self.z_TRAIN = z_calc(self.train_pat, self.train_sub_pat)
         self.z_starttrain, self.z_endtrain = calculating_slicesofinterest_segtrain(self.train_pat, self.train_sub_pat, self.z_TRAIN, self.roi_interest)
         self.z_starttrain, self.z_endtrain = calculating_slicesofinterestfor3D(self.train_pat,self.z_starttrain, self.z_endtrain)
         self.x_centtrain, self.y_centtrain = calculate_centroid_segtrain(self.train_pat,
             self.train_sub_pat, self.z_starttrain, self.z_endtrain, self.roi_interest)
         np.save(os.path.join(self.main_pred_dir, 'train.npy'), self.train_pat)
         np.save(os.path.join(self.main_pred_dir, 'train_sub.npy'), self.train_sub_pat)
         np.save(os.path.join(self.main_pred_dir, 'z_train.npy'), self.z_TRAIN)
         np.save(os.path.join(self.main_pred_dir, 'z_starttrain.npy'), self.z_startvalid)
         np.save(os.path.join(self.main_pred_dir, 'z_endtrain.npy'), self.z_endtrain)
         np.save(os.path.join(self.main_pred_dir, 'x_centtrain.npy'), self.x_centtrain)
         np.save(os.path.join(self.main_pred_dir, 'y_centtrain.npy'), self.y_centtrain)

    def preprocessing_valid(self):
         self.z_valid = z_calc(self.valid_pat, self.valid_sub_pat)
         self.z_startvalid, self.z_endvalid = calculating_slicesofinterest_segtrain(self.valid_pat, self.valid_sub_pat, self.z_valid, self.roi_interest)
         self.z_startvalid, self.z_endvalid = calculating_slicesofinterestfor3D(self.valid_pat,self.z_startvalid, self.z_endvalid)
         self.x_centvalid, self.y_centvalid = calculate_centroid_segvalid(self.valid_pat,
             self.valid_sub_pat, self.z_startvalid, self.z_endvalid, self.roi_interest)
         np.save(os.path.join(self.main_pred_dir, 'valid.npy'), self.valid_pat)
         np.save(os.path.join(self.main_pred_dir, 'valid_sub.npy'), self.valid_sub_pat)
         np.save(os.path.join(self.main_pred_dir, 'z_valid.npy'), self.z_valid)
         np.save(os.path.join(self.main_pred_dir, 'z_startvalid.npy'), self.z_startvalid)
         np.save(os.path.join(self.main_pred_dir, 'z_endvalid.npy'), self.z_endvalid)
         np.save(os.path.join(self.main_pred_dir, 'x_centvalid.npy'), self.x_centvalid)
         np.save(os.path.join(self.main_pred_dir, 'y_centvalid.npy'), self.y_centvalid)

    def traindatagenerator(self):
        while True:
            # Randomize the indices to make an array
            indices_arr = np.random.permutation(len(self.train_pat))

            for batch in range(0, len(indices_arr), self.batch_size):
                #print(batch, n)
                # slice out the current batch according to batch-size
                current_batch = indices_arr[batch:(batch + self.batch_size)]
                # initializing the arrays, x_train and y_train
                x_train = np.ndarray((self.batch_size, self.img_rows, self.img_cols, self.img_slcs, 1), dtype=np.float32)
                y_train = np.ndarray((self.batch_size, self.img_rows, self.img_cols, self.img_slcs, 1), dtype=np.float32)
                n = 0
                for i in current_batch:
                    x1 = self.train_pat[i]
                    t1 = self.train_pat_sublist[i]
                    xof1 = self.x_centtrain[i] - 80
                    xof2 = self.x_centtrain[i] + 80
                    yof1 = self.y_centtrain[i] - 80
                    yof2 = self.y_centtrain[i] + 80
                    if yof2 > 512:
                        yof1 = 352
                        yof2 = 512
                    train_img_in = np.load(os.path.join(self.data_dir, "CT_{}_{}.npy".format(x1, t1)))[xof1:xof2,
                                   yof1:yof2,
                                   self.z_starttrain[i]:self.z_endtrain[i]]
                    train_img_roi = np.load(os.path.join(self.data_dir, "ROI_{}_{}_{}.npy".format(x1, t1, self.roi_interest)))[xof1:xof2,
                                    yof1:yof2, self.z_starttrain[i]:self.z_endtrain[i]]

                    train_img_in_equalized = np.zeros(train_img_in.shape)
                    inverse_image = np.ones(train_img_roi.shape)
                    for r in range(train_img_in.shape[2]):
                        imagenew = train_img_in[:, :, r]
                        train_img_in_equalized[:, :, r] = image_histogram_equalization(imagenew)[0]

                    # Appending them to existing batch
                    x_train[n:(n + 1),:,:,:,0] = train_img_in_equalized/255
                    y_train[n:(n + 1),:,:,:,0] = train_img_roi
                    #print(batch, i, np.max(train_img_dist),np.max(y_train2))
                    n += 1
                yield (x_train, y_train)

    def validdatagenerator(self):
        while True:
            # Randomize the indices to make an array
            indices_arr = np.random.permutation(len(self.valid_pat))

            for batch in range(0, len(indices_arr), self.batch_size):
                # slice out the current batch according to batch-size
                current_batch = indices_arr[batch:(batch + self.batch_size)]
                # initializing the arrays, x_train and y_train
                n = 0
                x_valid  = np.ndarray((self.batch_size, self.img_rows, self.img_cols, self.img_slcs, 1), dtype=np.float32)
                y_valid = np.ndarray((self.batch_size, self.img_rows, self.img_cols, self.img_slcs, 1), dtype=np.float32)
                for i in current_batch:
                    x1 = self.valid_pat[i]
                    t1 = self.valid_pat_sublist[i]
                    xof1 = self.x_centvalid[i] - 80
                    xof2 = self.x_centvalid[i] + 80
                    yof1 = self.y_centvalid[i] - 80
                    yof2 = self.y_centvalid[i] + 80
                    if yof2 > 512:
                        yof1 = 352
                        yof2 = 512
                    valid_img_in = np.load(os.path.join(self.data_dir, "CT_{}_{}.npy".format(x1, t1)))[xof1:xof2,
                                   yof1:yof2,
                                   self.z_startvalid[i]:self.z_endvalid[i]]
                    valid_img_roi = np.load(os.path.join(self.data_dir, "ROI_{}_{}_{}.npy".format(x1, t1, self.roi_interest)))[xof1:xof2,
                                    yof1:yof2, self.z_startvalid[i]:self.z_endvalid[i]]

                    valid_img_in_equalized = np.zeros(valid_img_in.shape)
                    inverse_image = np.ones(valid_img_roi.shape)
                    for r in range(valid_img_in.shape[2]):
                        imagenew = valid_img_in[:, :, r]
                        valid_img_in_equalized[:, :, r] = image_histogram_equalization(imagenew)[0]
                    valid_img_dist = distance_transform_edt(inverse_image - valid_img_roi.astype('float32'))
                    # Appending them to existing batch
                    x_valid[n:(n + 1),:,:,:,0] = valid_img_in_equalized/255
                    y_valid[n:(n + 1),:,:,:,0] = valid_img_roi
                    n += 1
                yield (x_valid, y_valid)

    def train(self):
        self.preprocessing_train()
        self.preprocessing_valid()
        self.Rectum_network = unet_3D_ResNet_DB(self.img_rows, self.img_cols,self.img_slcs, channels_in=1, channels_out=1,
            starting_filter_number=self.starting_filter,kernelsize=(3, 3, 3), number_of_pool=self.number_of_pool, poolsize=self.poolsize,                                                            filter_rate=2, dropout_rate=self.dropout_rate,
            final_activation='sigmoid')
        self.Rectum_network.compile(optimizer=Adam(lr=1e-2), loss=weighted_dice_loss3D, metrics=[dice_coef])
        model_checkpoint = ModelCheckpoint('Rectum_weights.h5', monitor='val_dice_coef', save_best_only=True, mode='max')
        self.Rectum_network.fit(self.traindatagenerator, batch_size=self.batch_size, epochs= self.epochs, verbose=1, shuffle=True,
                                             validation_data = self.validdatagenerator,callbacks=[model_checkpoint])
        self.Rectum_network.save('Rectum_network.h5')

if __name__ == '__main__':
    Rectum_model = Rectum_segmentation()
    Rectum_model.train()
















