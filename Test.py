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
from src.Utils.LossFunctions import weighted_dice_loss3D, dice_coef, channel_RSDice, channel_RSloss
from src.Utils.Data_Utils import z_calc,calculating_slicesofinterest_segvalid, \
    calculating_slicesofinterestfor3D, calculate_centroid_segtrain, calculate_centroid_segvalid, image_histogram_equalization, save_coarse_predictions
tensorflow.compat.v1.disable_eager_execution()
from numpy import asarray
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from src.Models.DropBlock import DropBlock3D
from src.Models.groupnorm import GroupNormalization
from scipy import ndimage

class postop_testing():

    def __init__(self):
        print("-------------------------------------------------------------------------------------------------------")
        print("INITIALIZING VARIABLES")
        print("-------------------------------------------------------------------------------------------------------")
        self.data_dir = '/data/CTVSeg'
        self.loc_dir = '/data/localization'

    def create_locnpys(self, train_pat, sub, z_train):
        data_list1 = list()
        data_list2 = list()
        data_list3 = list()
        data_list4 = list()
        data_list5 = list()
        data_list6 = list()
        data_list8 = list()

        for i in range(len(train_pat)):
            p = train_pat[i]
            s = sub[i]
            print(p)
            ct = np.load(self.data_dir + 'CT_{}_{}.npy'.format(p, s))
            roi2 = np.load(self.data_dir + 'ROI_{}_{}_2.npy'.format(p, s))
            roi3 = np.load(self.data_dir + 'ROI_{}_{}_3.npy'.format(p, s))
            roi4 = np.load(self.data_dir + 'ROI_{}_{}_4.npy'.format(p, s))
            roi5 = np.load(self.data_dir + 'ROI_{}_{}_5.npy'.format(p, s))
            roi6 = np.load(self.data_dir + 'ROI_{}_{}_7.npy'.format(p, s))
            roi8 = np.load(self.data_dir + 'ROI_{}_{}_8.npy'.format(p, s))
            for z in range(z_train[i]):
                data_list1.append(ct[:, :, z].reshape(512, 512, 1))
                data_list2.append(roi2[:, :, z].reshape(512, 512, 1))
                data_list3.append(roi3[:, :, z].reshape(512, 512, 1))
                data_list4.append(roi4[:, :, z].reshape(512, 512, 1))
                data_list5.append(roi5[:, :, z].reshape(512, 512, 1))
                data_list6.append(roi6[:, :, z].reshape(512, 512, 1))
                data_list8.append(roi8[:, :, z].reshape(512, 512, 1))
        X1 = asarray(data_list1)
        X2 = asarray(data_list2)
        X3 = asarray(data_list3)
        X4 = asarray(data_list4)
        X5 = asarray(data_list5)
        X6 = asarray(data_list6)
        X7 = asarray(data_list8)

        X1 = np.tanh(X1[:,100:420,100:420, :] / 1000)
        Xroi = np.ndarray((X1.shape[0], X1.shape[1], X1.shape[2], 4), dtype=np.float32)
        Xroi[:, :, :, 0:1] = X2[:,100:420,100:420, :] + X3[:,100:420,100:420, :]
        Xroi[:, :, :, 1:2] = X4[:,100:420,100:420, :]
        Xroi[:, :, :, 2:3] = X5[:,100:420,100:420, :]
        Xroi[:, :, :, 3:4] = X6[:,100:420,100:420, :]
        Xroi[:, :, :, 4:5] = X7[:,100:420,100:420, :]
        return X1, Xroi

    def preprocessing_test(self, oar_pred_dir, oar):
         self.z_starttest, self.z_endtest = calculating_slicesofinterest_segvalid(self.test_pat, self.test_sub, self.z_test, oar)
         self.z_starttest, self.z_endtest = calculating_slicesofinterestfor3D(self.test_pat,self.z_starttest, self.z_endtest)
         self.x_centtest, self.y_centtest = calculate_centroid_segvalid(self.test_pat, self.test_sub, self.z_starttest, self.z_endtest, oar)
         np.save(os.path.join(oar_pred_dir, 'test.npy'), self.test_pat)
         np.save(os.path.join(oar_pred_dir, 'test_sub.npy'), self.test_sub)
         np.save(os.path.join(oar_pred_dir, 'z_test.npy'), self.z_test)
         np.save(os.path.join(oar_pred_dir, 'z_starttest.npy'), self.z_starttest)
         np.save(os.path.join(oar_pred_dir, 'z_endtest.npy'), self.z_endtest)
         np.save(os.path.join(oar_pred_dir, 'x_centtest.npy'), self.x_centtest)
         np.save(os.path.join(oar_pred_dir, 'y_centtest.npy'), self.y_centtest)

    def predict_with_uncertainty(self, oar_dir, oar, f, x, no_classes, n_iter=2):
        result = np.zeros((n_iter,) + (x.shape[0], x.shape[1], x.shape[2], no_classes))
        pred = np.zeros((len(x), 160, 160, 64))
        for i in range(n_iter):
            pred[i] = f((x, 1))[0]
        Pred_Mean = np.mean(pred, axis=0) > .7
        Pred_var = np.std(pred, axis=0).flatten()
        pred_flat = Pred_Mean.flatten()
        pred_flat_up = Pred_Mean.flatten()
        pred_flat_down = Pred_Mean.flatten()
        for j in range(len(pred_flat)):
            pred_flat_up[j] = pred_flat[j] + 2 * Pred_var[j]
            pred_flat_down[j] = pred_flat[j] - 2 * Pred_var[j]
        bounds_high = pred_flat_up.reshape((160, 160, 64)) > 0
        bounds_low = pred_flat_down.reshape((160, 160, 64)) == 1.0
        Pred_Mean[i, :, :, :] = ndimage.binary_fill_holes(Pred_Mean[i, :, :, :])
        bounds_high = ndimage.binary_fill_holes(bounds_high)
        bounds_low = ndimage.binary_fill_holes(bounds_low)
        np.save(os.path.join(oar_dir,'ROI_{}_{}_Mean'))



    def predict(self):
        self.test_pat = np.load(os.path.join(self.data_dir, 'test.npy'))
        self.test_sub = np.load(os.path.join(self.data_dir, 'test_sub.npy'))
        self.z_test = z_calc(self.test_pat, self.test_sub)

        #localization
        self.test_imgs_in, self.test_imgs_roi = self.create_locnpys(self.test_pat, self.test_sub,self.z_test)
        cust = {'channel_RSDice': channel_RSDice, 'channel_RSloss': channel_RSloss}
        self.loc_network = load_model('locnetwork_2.h5', cust)
        imgs_roi_pred = self.loc_network.predict(self.test_imgs_in, batch_size=1, verbose=1)
        save_coarse_predictions(self.loc_dir, imgs_roi_pred)

        #OAR segmentation
        oar_pred_dir = '/data/M1_FM2'
        self.preprocessing_test(oar_pred_dir, 2)
        cust = {'weighted_dice_loss3D': weighted_dice_loss3D, 'dice_coef': dice_coef, 'GroupNormalization':GroupNormalization, 'DropBlock3D':DropBlock3D}
        self.loc_network = load_model('FM2_network.h5', cust)
        f = K.function([self.loc_network.layers[0].input, K.learning_phase()], [self.loc_network.layers[-1].output])
        self.predict_with_uncertainty(f, oar_pred_dir, 5, self.test_imgs_in, 1, n_iter=50)

        oar_pred_dir = '/data/M1_FM3'
        self.preprocessing_test(oar_pred_dir, 3)
        cust = {'weighted_dice_loss3D': weighted_dice_loss3D, 'dice_coef': dice_coef, 'GroupNormalization':GroupNormalization, 'DropBlock3D':DropBlock3D}
        self.loc_network = load_model('FM3_network.h5', cust)
        f = K.function([self.loc_network.layers[0].input, K.learning_phase()], [self.loc_network.layers[-1].output])
        self.predict_with_uncertainty(f, oar_pred_dir, 5, self.test_imgs_in, 1, n_iter=50)
        oar_pred_dir = '/data/M1_PB'
        self.preprocessing_test(oar_pred_dir, 8)
        cust = {'weighted_dice_loss3D': weighted_dice_loss3D, 'dice_coef': dice_coef, 'GroupNormalization':GroupNormalization, 'DropBlock3D':DropBlock3D}
        self.loc_network = load_model('PB_network.h5', cust)
        f = K.function([self.loc_network.layers[0].input, K.learning_phase()], [self.loc_network.layers[-1].output])
        self.predict_with_uncertainty(f, oar_pred_dir, 5, self.test_imgs_in, 1, n_iter=50)

        oar_pred_dir = '/data/M1_Bladder'
        self.preprocessing_test(oar_pred_dir, 5)
        cust = {'weighted_dice_loss3D': weighted_dice_loss3D, 'dice_coef': dice_coef, 'GroupNormalization':GroupNormalization, 'DropBlock3D':DropBlock3D}
        self.loc_network = load_model('Bladder_network.h5', cust)
        f = K.function([self.loc_network.layers[0].input, K.learning_phase()], [self.loc_network.layers[-1].output])
        self.predict_with_uncertainty(f, oar_pred_dir, 5, self.test_imgs_in, 1, n_iter=50)
        oar_pred_dir = '/data/M1_Rectum'
        self.preprocessing_test(oar_pred_dir, 4)
        cust = {'weighted_dice_loss3D': weighted_dice_loss3D, 'dice_coef': dice_coef, 'GroupNormalization':GroupNormalization, 'DropBlock3D':DropBlock3D}
        self.loc_network = load_model('Rectum_network.h5', cust)
        f = K.function([self.loc_network.layers[0].input, K.learning_phase()], [self.loc_network.layers[-1].output])
        self.predict_with_uncertainty(f, oar_pred_dir, 4, self.test_imgs_in, 1, n_iter=50)
        oar_pred_dir = '/data/M1_CTV'
        self.preprocessing_test(oar_pred_dir, 7)
        cust = {'weighted_dice_loss3D': weighted_dice_loss3D, 'dice_coef': dice_coef, 'GroupNormalization':GroupNormalization, 'DropBlock3D':DropBlock3D}
        self.loc_network = load_model('Rectum_network.h5', cust)
        f = K.function([self.loc_network.layers[0].input, K.learning_phase()], [self.loc_network.layers[-4].output])
        self.predict_with_uncertainty(f, oar_pred_dir, 7, self.test_imgs_in, 1, n_iter=50)




if __name__ == '__main__':
    test_model = postop_testing()
    test_model.predict()
