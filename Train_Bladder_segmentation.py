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
from src.Models.SegmentationNetworks import unet_3D_ResNeXt_DB
from src.Utils.LossFunctions import weighted_dice_loss3D, dice_coef
from src.Utils.Data_Utils import z_calc,calculating_slicesofinterest_segtrain, \
    calculating_slicesofinterestfor3D, calculate_centroid_segtrain, calculate_centroid_segvalid, image_histogram_equalization

class Bladder_segmentation():

    def __init__(self):
        self.valid_pat_sublist = None
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
        self.data_dir = '/data/BED_OAR_MASKS'
        self.pat = np.load('/data/CTVSeg/train_pat_postop.npy')
        split = int(.9*len(self.pat))
        end = len(self.pat)
        self.train_pat = self.pat[0:split]
        print('train total', len(self.train_pat))
        self.train_pat_sublist = np.load('/data/s426200/CTVSeg/train_sub_pat_postop.npy')[0:split]
        self.valid_pat = self.pat[split:end]
        self.valid_sub_pat = np.load('/data/train_sub_pat_postop.npy')[split:end]
        print('valid total',len(self.valid_pat))
        self.main_pred_dir = '/data/M1_Bladder'
        self.roi_interest = 5

    def preprocessing_train(self):
    #      self.z_train = z_calc(self.train_pat, self.train_pat_sublist)
    #      self.z_starttrain, self.z_endtrain = calculating_slicesofinterest_segtrain(self.train_pat, self.train_pat_sublist, self.z_train, self.roi_interest)
    #      self.z_starttrain, self.z_endtrain = calculating_slicesofinterestfor3D(self.train_pat,self.z_starttrain, self.z_endtrain, self.z_train, 64)
    #      self.x_centtrain, self.y_centtrain = calculate_centroid_segtrain(self.train_pat,
    #          self.train_pat_sublist, self.z_starttrain, self.z_endtrain, self.roi_interest)
    #      np.save(os.path.join(self.main_pred_dir, 'train.npy'), self.train_pat)
    #      np.save(os.path.join(self.main_pred_dir, 'train_sub.npy'), self.train_pat_sublist)
    #      np.save(os.path.join(self.main_pred_dir, 'z_train.npy'), self.z_train)
    #      np.save(os.path.join(self.main_pred_dir, 'z_starttrain.npy'), self.z_starttrain)
    #      np.save(os.path.join(self.main_pred_dir, 'z_endtrain.npy'), self.z_endtrain)
    #      np.save(os.path.join(self.main_pred_dir, 'x_centtrain.npy'), self.x_centtrain)
    #      np.save(os.path.join(self.main_pred_dir, 'y_centtrain.npy'), self.y_centtrain)
         self.train_pat = np.load(os.path.join(self.main_pred_dir, 'train.npy'))
         self.train_pat_sublist = np.load(os.path.join(self.main_pred_dir, 'train_sub.npy'))
         self.z_train = np.load(os.path.join(self.main_pred_dir, 'z_train.npy'))
         self.z_starttrain = np.load(os.path.join(self.main_pred_dir, 'z_starttrain.npy'))
         self.z_endtrain =  np.load(os.path.join(self.main_pred_dir, 'z_endtrain.npy'))
         self.x_centtrain = np.load(os.path.join(self.main_pred_dir, 'x_centtrain.npy'))
         self.y_centtrain = np.load(os.path.join(self.main_pred_dir, 'y_centtrain.npy'))

    def preprocessing_valid(self):
         # self.z_valid = z_calc(self.valid_pat, self.valid_sub_pat)
         # print(len(self.valid_pat), len(self.valid_sub_pat), len(self.z_valid))
         # self.z_startvalid, self.z_endvalid = calculating_slicesofinterest_segtrain(self.valid_pat, self.valid_sub_pat, self.z_valid, self.roi_interest)
         # self.z_startvalid, self.z_endvalid = calculating_slicesofinterestfor3D(self.valid_pat,self.z_startvalid, self.z_endvalid, self.z_valid, 64)
         # self.x_centvalid, self.y_centvalid = calculate_centroid_segtrain(self.valid_pat,
         #     self.valid_sub_pat, self.z_startvalid, self.z_endvalid, self.roi_interest)
         np.save(os.path.join(self.main_pred_dir, 'valid.npy'), self.valid_pat)
         # np.save(os.path.join(self.main_pred_dir, 'valid_sub.npy'), self.valid_sub_pat)
         # np.save(os.path.join(self.main_pred_dir, 'z_valid.npy'), self.z_valid)
         # np.save(os.path.join(self.main_pred_dir, 'z_startvalid.npy'), self.z_startvalid)
         # np.save(os.path.join(self.main_pred_dir, 'z_endvalid.npy'), self.z_endvalid)
         # np.save(os.path.join(self.main_pred_dir, 'x_centvalid.npy'), self.x_centvalid)
         # np.save(os.path.join(self.main_pred_dir, 'y_centvalid.npy'), self.y_centvalid)
         self.valid_pat = np.load(os.path.join(self.main_pred_dir, 'valid.npy'))
         self.valid_sub_pat = np.load(os.path.join(self.main_pred_dir, 'valid_sub.npy'))
         self.z_valid = np.load(os.path.join(self.main_pred_dir, 'z_valid.npy'))
         self.z_startvalid = np.load(os.path.join(self.main_pred_dir, 'z_startvalid.npy'))
         self.z_endvalid = np.load(os.path.join(self.main_pred_dir, 'z_endvalid.npy'))
         self.x_centvalid = np.load(os.path.join(self.main_pred_dir, 'x_centvalid.npy'))
         self.y_centvalid = np.load(os.path.join(self.main_pred_dir, 'y_centvalid.npy'))

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
                    x_train[n,:,:,:,0] = train_img_in_equalized/255
                    y_train[n,:,:,:,0] = train_img_roi
                    #print(batch, i, np.max(train_img_dist),np.max(y_train2))
                    n += 1
                yield x_train, y_train

    def validdatagenerator(self):
        while True:
            # Randomize the indices to make an array
            indices_arr = np.array(range(len(self.valid_pat)))
            for batch in range(0, len(indices_arr), self.batch_size):
                #print(batch, n)
                # slice out the current batch according to batch-size
                current_batch = indices_arr[batch:(batch + self.batch_size)]
                # initializing the arrays, x_train and y_train
                x_valid = np.ndarray((self.batch_size, self.img_rows, self.img_cols, self.img_slcs, 1), dtype=np.float32)
                y_valid = np.ndarray((self.batch_size, self.img_rows, self.img_cols, self.img_slcs, 1), dtype=np.float32)
                n = 0
                for i in current_batch:
                    x1 = self.valid_pat[i]
                    t1 = self.valid_sub_pat[i]
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
                    for r in range(valid_img_in.shape[2]):
                        imagenew = valid_img_in[:, :, r]
                        valid_img_in_equalized[:, :, r] = image_histogram_equalization(imagenew)[0]

                    # Appending them to existing batch
                    x_valid[n,:,:,:,0] = valid_img_in_equalized/255
                    y_valid[n,:,:,:,0] = valid_img_roi
                    #print(batch, i, np.max(train_img_dist),np.max(y_train2))
                    n += 1
                yield x_valid, y_valid

    def train(self):
        self.preprocessing_train()
        self.preprocessing_valid()
        for i in range(len(self.valid_pat)):
            print(self.valid_pat[i], self.valid_sub_pat[i])
        self.Bladder_network = unet_3D_ResNeXt_DB(self.img_rows, self.img_cols,self.img_slcs, cardinality=16,channels_in=1, channels_out=1,
            starting_filter_number=self.starting_filter,kernelsize=(3, 3, 3), number_of_pool=self.number_of_pool, poolsize=self.poolsize,filter_rate=2,
                                                  final_activation='sigmoid')
        self.Bladder_network.compile(optimizer=Adam(lr=1e-2), loss=weighted_dice_loss3D, metrics=[dice_coef])
        model_checkpoint = ModelCheckpoint('Bladder_weights.h5', monitor='val_dice_coef', save_best_only=True, mode='max')
        self.Bladder_network.fit(self.traindatagenerator(), steps_per_epoch=int(self.epochs/self.batch_size), epochs= self.epochs, verbose=1,
                                             validation_data = self.validdatagenerator(), callbacks=[model_checkpoint])
        self.Bladder_network.save('Bladder_network.h5')

if __name__ == '__main__':
    Bladder_model = Bladder_segmentation()
    Bladder_model.train()
















