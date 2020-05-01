import os
import numpy as np
import random
from src.Utils.Data_Utils import image_histogram_equalization

data_dir = '/data/mnt/share/dan/edrive/Anjali_Backup/BED_OAR_MASKS'
# data_dir = '/data/mnt/share/dan/edrive/Anjali_Backup/NUMPYS'
# data_dir = "/data/s426200/NUMPYS"

def createdatatensor(seg_dir, img_rows,  img_cols, img_slcs, pat, sub, z_start, z_end, x_cent, y_cent, roi_interest, trainorvalid, justload):
    if justload:
        if trainorvalid == 't':
            x_train = np.load(os.path.join(seg_dir, 'train_in.npy'))
            y_train = np.load(os.path.join(seg_dir, 'train_out.npy'))
        if trainorvalid == 'v':
            x_train = np.load(os.path.join(seg_dir, 'valid_in.npy'))
            y_train = np.load(os.path.join(seg_dir, 'valid_out.npy'))
    else:
        x_train = np.ndarray((len(pat), img_rows, img_cols, img_slcs, 1), dtype=np.float32)
        y_train = np.ndarray((len(pat), img_rows, img_cols, img_slcs, 1), dtype=np.float32)
        for i in range(len(pat)):
            x1 = pat[i]
            t1 = sub[i]
            xof1 = x_cent[i] - int(img_rows/2)
            xof2 = x_cent[i] + int(img_rows/2)
            yof1 = y_cent[i] - int(img_cols/2)
            yof2 = y_cent[i] + int(img_cols/2)
            if yof2 > 512:
                yof1 = 352
                yof2 = 512
            train_img_in = np.load(os.path.join(data_dir, "CT_{}_{}.npy".format(x1, t1)))[xof1:xof2, yof1:yof2,z_start[i]:z_end[i]]
            train_img_roi = np.load(os.path.join(data_dir, "ROI_{}_{}_{}.npy".format(x1, t1, roi_interest)))[xof1:xof2, yof1:yof2, z_start[i]:z_end[i]]
            train_img_in_equalized = np.zeros(train_img_in.shape)
            for r in range(train_img_in.shape[2]):
                imagenew = train_img_in[:, :, r]
                train_img_in_equalized[:, :, r] = image_histogram_equalization(imagenew)[0]
            # Appending them to existing batch
            x_train[i, :, :, :, 0] = train_img_in_equalized / 255
            y_train[i, :, :, :, 0] = train_img_roi

        if trainorvalid == 't':
           np.save(os.path.join(seg_dir, 'train_in.npy'),  x_train)
           np.save(os.path.join(seg_dir, 'train_out.npy'), y_train)
           # print('T done')
        if trainorvalid == 'v':
           np.save(os.path.join(seg_dir, 'valid_in.npy'),  x_train)
           np.save(os.path.join(seg_dir, 'valid_out.npy'), y_train)
           # print('V done')
    return x_train, y_train

def createdatatensorwithaugmentation(seg_dir, img_rows,  img_cols, img_slcs, pat, sub, z_start, z_end, x_cent, y_cent, roi_interest, trainorvalid, justload):
    if justload:
        if trainorvalid == 't':
            x_train = np.load(os.path.join(seg_dir, 'train_in.npy'))
            y_train = np.load(os.path.join(seg_dir, 'train_out.npy'))
        if trainorvalid == 'v':
            x_train = np.load(os.path.join(seg_dir, 'valid_in.npy'))
            y_train = np.load(os.path.join(seg_dir, 'valid_out.npy'))
    else:
        if trainorvalid == 't':
            x_train = np.ndarray((len(pat)*2, img_rows, img_cols, img_slcs, 1), dtype=np.float32)
            y_train = np.ndarray((len(pat)*2, img_rows, img_cols, img_slcs, 1), dtype=np.float32)
            aug = random.randrange(1,4)
        else:
            x_train = np.ndarray((len(pat), img_rows, img_cols, img_slcs, 1), dtype=np.float32)
            y_train = np.ndarray((len(pat), img_rows, img_cols, img_slcs, 1), dtype=np.float32)
            aug = 0
        n = 0
        for i in range(len(pat)):
            x1 = pat[i]
            t1 = sub[i]
            xof1 = x_cent[i] - int(img_rows/2)
            xof2 = x_cent[i] + int(img_rows/2)
            yof1 = y_cent[i] - int(img_cols/2)
            yof2 = y_cent[i] + int(img_cols/2)
            if yof2 > 512:
                yof1 = 352
                yof2 = 512
            train_img_in = np.load(os.path.join(data_dir, "CT_{}_{}.npy".format(x1, t1)))[xof1:xof2, yof1:yof2,z_start[i]:z_end[i]]
            train_img_roi = np.load(os.path.join(data_dir, "ROI_{}_{}_{}.npy".format(x1, t1, roi_interest)))[xof1:xof2, yof1:yof2, z_start[i]:z_end[i]]
            train_img_in_equalized = np.zeros(train_img_in.shape)
            for r in range(train_img_in.shape[2]):
                imagenew = train_img_in[:, :, r]
                train_img_in_equalized[:, :, r] = image_histogram_equalization(imagenew)[0]
            # Appending them to existing batch
            x_train[n, :, :, :, 0] = train_img_in_equalized / 255
            y_train[n, :, :, :, 0] = train_img_roi
            n += 1
            if aug:
                if aug == 1:
                   train_img_in = np.load(os.path.join(data_dir, "CT_{}_{}.npy".format(x1, t1)))[xof1:xof2, yof1:yof2,
                                   z_start[i]-8:z_end[i]-8]
                   train_img_roi = np.load(os.path.join(data_dir, "ROI_{}_{}_{}.npy".format(x1, t1, roi_interest)))[xof1:xof2, yof1:yof2,
                                   z_start[i]-8:z_end[i]-8]
                if aug == 2:
                   train_img_in = np.load(os.path.join(data_dir, "CT_{}_{}.npy".format(x1, t1)))[xof1:xof2, yof1:yof2,
                                   z_start[i]+8:z_end[i]+8]
                   train_img_roi = np.load(os.path.join(data_dir, "ROI_{}_{}_{}.npy".format(x1, t1, roi_interest)))[xof1:xof2, yof1:yof2,
                                   z_start[i]+8:z_end[i]+8]
                if aug == 3:
                   train_img_in = np.load(os.path.join(data_dir, "CT_{}_{}.npy".format(x1, t1)))[xof1:xof2, yof1:yof2,
                                   z_start[i]:z_end[i]]
                   train_img_roi = np.load(os.path.join(data_dir, "ROI_{}_{}_{}.npy".format(x1, t1, roi_interest)))[xof1:xof2, yof1:yof2,
                                   z_start[i]:z_end[i]]
                train_img_in_equalized = np.zeros(train_img_in.shape)
                for r in range(train_img_in.shape[2]):
                    imagenew = train_img_in[:, :, r]
                    train_img_in_equalized[:, :, r] = image_histogram_equalization(imagenew)[0]
                # Appending them to existing batch
                x_train[n+1, :, :, :, 0] = train_img_in_equalized / 255
                y_train[n+1, :, :, :, 0] = train_img_roi
                n += 1
        if trainorvalid == 't':
           np.save(os.path.join(seg_dir, 'train_in.npy'),  x_train)
           np.save(os.path.join(seg_dir, 'train_out.npy'), y_train)
           # print('T done')
        if trainorvalid == 'v':
           np.save(os.path.join(seg_dir, 'valid_in.npy'),  x_train)
           np.save(os.path.join(seg_dir, 'valid_out.npy'), y_train)
           # print('V done')
    return x_train, y_train

def datagenerator(batch_size, img_rows,  img_cols, img_slcs, pat, sub, z_start, z_end, x_cent, y_cent, roi_interest, trainorvalid):
    while True:
        # Randomize the indices to make an array if training
        if trainorvalid == 't':
           indices_arr = np.random.permutation(len(pat))
        else:
           indices_arr = np.array(range(len(pat)))
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]
            # initializing the arrays, x_train and y_train
            x_train = np.ndarray((batch_size, img_rows, img_cols, img_slcs, 1), dtype=np.float32)
            y_train = np.ndarray((batch_size, img_rows, img_cols, img_slcs, 1), dtype=np.float32)
            n = 0
            for i in current_batch:
                x1 = pat[i]
                t1 = sub[i]
                xof1 = x_cent[i] - 80
                xof2 = x_cent[i] + 80
                yof1 = y_cent[i] - 80
                yof2 = y_cent[i] + 80
                if yof2 > 512:
                    yof1 = 352
                    yof2 = 512
                train_img_in = np.load(os.path.join(data_dir, "CT_{}_{}.npy".format(x1, t1)))[xof1:xof2,
                               yof1:yof2,z_start[i]:z_end[i]]
                train_img_roi = np.load(os.path.join(data_dir, "ROI_{}_{}_{}.npy".format(x1, t1, roi_interest)))[xof1:xof2,
                                yof1:yof2, z_start[i]:z_end[i]]

                train_img_in_equalized = np.zeros(train_img_in.shape)
                for r in range(train_img_in.shape[2]):
                    imagenew = train_img_in[:, :, r]
                    train_img_in_equalized[:, :, r] = image_histogram_equalization(imagenew)[0]

                # Appending them to existing batch
                x_train[n,:,:,:,0] = train_img_in_equalized/255
                y_train[n,:,:,:,0] = train_img_roi
                n += 1
            yield x_train, y_train

def createlocalizationtensor_validtest(loc_dir, pat, sub, z_in):
    n_dim = 0
    for s in z_in:
        n_dim += s
    x_train = np.ndarray((n_dim, 256, 320, 1), dtype=np.float32)
    y_train = np.ndarray((n_dim, 256, 320, 4), dtype=np.float32)
    z = 0
    for i in range(len(pat)):
        x1 = pat[i]
        t1 = sub[i]
        img_in = np.load(os.path.join(data_dir, "CT_{}_{}.npy".format(x1, t1)))[124:380, 100:420,:]
        try:
            mask_out0 = np.load(os.path.join(data_dir, "ROI_{}_{}_2.npy".format(x1, t1)))[124:380, 100:420,:] + np.load(os.path.join(data_dir, "ROI_{}_{}_3.npy".format(x1, t1)))[124:380, 100:420,:]
        except:
            mask_out0 = np.zeros((256,320,z_in[i]))
        mask_out1 = np.load(os.path.join(data_dir, "ROI_{}_{}_4.npy".format(x1, t1)))[124:380, 100:420,:]
        mask_out2 = np.load(os.path.join(data_dir, "ROI_{}_{}_5.npy".format(x1, t1)))[124:380, 100:420,:]
        try:
            mask_out3 = np.load(os.path.join(data_dir, "ROI_{}_{}_7.npy".format(x1, t1)))[124:380, 100:420,:]
        except:
            mask_out3 = np.load(os.path.join(data_dir, "ROI_{}_{}_7_Phy.npy".format(x1, t1)))[124:380, 100:420,:]
        x_train[z:z+z_in[i], :, :, 0] = np.tanh(np.transpose(img_in/1000,(2,0,1)))
        y_train[z:z+z_in[i], :, :, 0] = np.transpose((mask_out0),(2,0,1))
        y_train[z:z+z_in[i], :, :, 1] = np.transpose((mask_out1),(2,0,1))
        y_train[z:z+z_in[i], :, :, 2] = np.transpose((mask_out2),(2,0,1))
        y_train[z:z+z_in[i], :, :, 3] = np.transpose((mask_out3),(2,0,1))
        z += z_in[i]
    np.save(os.path.join(loc_dir, 'loc_test_in.npy'), x_train)
    np.save(os.path.join(loc_dir, 'loc_test_out.npy'), y_train)
    return x_train, y_train



