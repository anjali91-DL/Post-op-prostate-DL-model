import os
import numpy as np
from numpy import asarray
from numpy import savez_compressed
import Data_Utils

npz_path = '/data/Localization'
main_pred_dir = '/data/Localization'
data_dir = '/data'

def load_images_valid_CT(path, train_pat, sub, z_train):
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
        ct = np.load(path + 'CT_{}_{}.npy'.format(p, s))
        roi2 = np.load(path + 'ROI_{}_{}_2.npy'.format(p, s))
        roi3 = np.load(path + 'ROI_{}_{}_3.npy'.format(p, s))
        roi4 = np.load(path + 'ROI_{}_{}_4.npy'.format(p, s))
        roi5 = np.load(path + 'ROI_{}_{}_5.npy'.format(p, s))
        roi6 = np.load(path + 'ROI_{}_{}_7.npy'.format(p, s))
        roi8 = np.load(path + 'ROI_{}_{}_8.npy'.format(p, s))
        for z in range(z_train[i]):
            data_list1.append(ct[:, :, z].reshape(512, 512, 1))
            data_list2.append(roi2[:, :, z].reshape(512, 512, 1))
            data_list3.append(roi3[:, :, z].reshape(512, 512, 1))
            data_list4.append(roi4[:, :, z].reshape(512, 512, 1))
            data_list5.append(roi5[:, :, z].reshape(512, 512, 1))
            data_list6.append(roi6[:, :, z].reshape(512, 512, 1))
            data_list8.append(roi8[:, :, z].reshape(512, 512, 1))
    return asarray(data_list1), asarray(data_list2), asarray(data_list3), asarray(data_list4), asarray(data_list5), asarray(data_list6), asarray(data_list8)

# dataset path
# # load valid/test data
valid_pat = np.load(os.path.join(main_pred_dir, 'valid.npy'))
valid_sub = np.load(os.path.join(main_pred_dir, 'valid_sub.npy'))
z_valid = Data_Utils.z_calc(valid_pat, valid_sub)
z_valid = np.save(os.path.join(main_pred_dir, 'z_valid.npy'),z_valid)
z_valid = np.load(os.path.join(main_pred_dir, 'z_valid.npy'))

roi_list = [2,3,4,5,7,8]

dataA1, dataA2, dataA3, dataA4, dataA5, dataA7 = load_images_valid_CT(data_dir, valid_pat, valid_sub, z_valid)

# save as compressed numpy array
filename = 'postopCT_valid_local.npz'
savez_compressed(npz_path + filename, dataA1, dataA2, dataA3, dataA4, dataA5, dataA7)
