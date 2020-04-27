import os
import numpy as np
from numpy import asarray
from numpy import savez_compressed

import Data_Utils

# define layer
npz_path = '/data/Localization'
data_dir = '/data/Localization'
main_pred_dir = '/data/Localization'

def load_images_train_CT(path, train_pat, train_sub, z_train, z_starttrain, z_endtrain):
    data_list1 = list()
    data_list2 = list()
    data_list3 = list()
    data_list4 = list()
    data_list5 = list()
    data_list6 = list()
    x = 0

    for i in range(len(train_pat)):
        p = train_pat[i]
        s = train_sub[i]
        ct = np.load(path + 'CT_{}_{}.npy'.format(p, s))
        roi2 = np.load(path + 'ROI_{}_{}_2.npy'.format(p, s))
        roi3 = np.load(path + 'ROI_{}_{}_3.npy'.format(p, s))
        roi4 = np.load(path + 'ROI_{}_{}_4.npy'.format(p, s))
        roi5 = np.load(path + 'ROI_{}_{}_5.npy'.format(p, s))
        roi6 = np.load(path + 'ROI_{}_{}_7.npy'.format(p, s))
        for z in range(z_starttrain[i], z_endtrain[i]):
            data_list1.append(ct[:, :, z].reshape(512, 512, 1))
            data_list2.append(roi2[:, :, z].reshape(512, 512, 1))
            data_list3.append(roi3[:, :, z].reshape(512, 512, 1))
            data_list4.append(roi4[:, :, z].reshape(512, 512, 1))
            data_list5.append(roi5[:, :, z].reshape(512, 512, 1))
            data_list6.append(roi6[:, :, z].reshape(512, 512, 1))
        if i % 2 == 0:
            for z in range(z_starttrain[i], z_endtrain[i]):
                data_list1.append(ct[:, :, z].reshape(512, 512, 1))
                data_list2.append(roi2[:, :, z].reshape(512, 512, 1))
                data_list3.append(roi3[:, :, z].reshape(512, 512, 1))
                data_list4.append(roi4[:, :, z].reshape(512, 512, 1))
                data_list5.append(roi5[:, :, z].reshape(512, 512, 1))
                data_list6.append(roi6[:, :, z].reshape(512, 512, 1))
                x += 1
        elif i % 5 == 0:
            for z in range(z_starttrain[i]):
                data_list1.append(ct[:, :, z].reshape(512, 512, 1))
                data_list2.append(roi2[:, :, z].reshape(512, 512, 1))
                data_list3.append(roi3[:, :, z].reshape(512, 512, 1))
                data_list4.append(roi4[:, :, z].reshape(512, 512, 1))
                data_list5.append(roi5[:, :, z].reshape(512, 512, 1))
                data_list6.append(roi6[:, :, z].reshape(512, 512, 1))
            x += 1
        elif i % 7 == 0:
            for z in range(z_endtrain[i], z_train[i]):
                data_list1.append(ct[:, :, z].reshape(512, 512, 1))
                data_list2.append(roi2[:, :, z].reshape(512, 512, 1))
                data_list3.append(roi3[:, :, z].reshape(512, 512, 1))
                data_list4.append(roi4[:, :, z].reshape(512, 512, 1))
                data_list5.append(roi5[:, :, z].reshape(512, 512, 1))
                data_list6.append(roi6[:, :, z].reshape(512, 512, 1))
            x += 1
        print(x)
    print(len(data_list1))
    return asarray(data_list1), asarray(data_list2), asarray(data_list3), asarray(data_list4), asarray(data_list5), asarray(data_list6)

train_pat = np.load(os.path.join(main_pred_dir, 'train.npy'))
train_sub = np.load(os.path.join(main_pred_dir, 'train_sub.npy'))
z_train = Data_Utils.z_calc(train_pat, train_sub)
Data_Utils.calculating_slicesofinterest(train_pat, train_sub, z_train)
z_starttrain, z_endtrain = Data_Utils.select_slices(main_pred_dir, train_pat)

dataA1, dataA2, dataA3, dataA4, dataA5, dataA7 = load_images_train_CT(data_dir, train_pat, train_sub, z_train, z_starttrain, z_endtrain)
print('Loaded dataA: ', dataA1.shape)

# print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = 'postopCT_train_local.npz'
savez_compressed(npz_path + filename, dataA1, dataA2, dataA3, dataA4, dataA5, dataA7)
