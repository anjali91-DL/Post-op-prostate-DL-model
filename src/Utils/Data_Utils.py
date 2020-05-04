import os
import numpy as np
from skimage.measure import label, regionprops


# data_dir = '/data/mnt/share/dan/edrive/Anjali_Backup/BED_OAR_MASKS'
# data_dir = '/data/mnt/share/dan/edrive/Anjali_Backup/NUMPYS'
data_dir = "/data/s426200/NUMPYS"

def z_calc(pat, pat_s):
    z_dim = []
    for i in range(0,len(pat)):
        p = pat[i]
        s = pat_s[i]
        #print(p, s)
        with open(os.path.join(data_dir, 'CT_{}_{}.npy'.format(p,s)), 'rb') as f:
            major, minor = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        z_dim.append(shape[2])
        np.save(os.path.join(data_dir, 'z_dim.npy'), z_dim)
    return z_dim

def calculating_slicesofinterest_trainloc(train_patlist,train_sublist, z_TRAIN):
    z_start = []  # [0]*len(TRAIN)
    z_end = [] #* len(train_patlist)
    for i1 in range(len(train_patlist)):
        x1 = train_patlist[i1]
        s1 = train_sublist[i1]
        fm2_mask = np.load(os.path.join(data_dir, 'ROI_{}_{}_{}.npy'.format(x1, s1, 2)))
        fm3_mask = np.load(os.path.join(data_dir, 'ROI_{}_{}_{}.npy'.format(x1, s1, 3)))
        rec_mask = np.load(os.path.join(data_dir, 'ROI_{}_{}_{}.npy'.format(x1, s1, 4)))
        blad_mask = np.load(os.path.join(data_dir, 'ROI_{}_{}_{}.npy'.format(x1, s1, 5)))
        ctv_mask = np.load(os.path.join(data_dir, 'ROI_{}_{}_{}.npy'.format(x1, s1, 7)))
        pb_mask = np.load(os.path.join(data_dir, 'ROI_{}_{}_{}.npy'.format(x1, s1, 7)))
        z_start1check = 0
        for i in range(z_TRAIN[i1]):
            if (z_start1check == 0 and (np.sum(rec_mask[:, :, i]) > 0 or np.sum(blad_mask[:, :, i]) > 0 or np.sum(
                    fm2_mask[:, :, i]) > 0 or np.sum(fm3_mask[:, :, i]) > 0 or np.sum(ctv_mask[:, :, i]) > 0)or np.sum(pb_mask[:, :, i]) > 0):
                z_start.append(i)
                z_start1check = 1
            if z_start1check == 1 and (np.sum(rec_mask[:, :, i]) == 0 and np.sum(blad_mask[:, :, i]) == 0 and np.sum(
                    fm2_mask[:, :, i]) == 0 and np.sum(fm3_mask[:, :, i]) == 0 and np.sum(ctv_mask[:, :, i]) == 0 and np.sum(pb_mask[:, :, i]) == 0):
                z_end.append(i)
                z_start1check = 2
        slices = z_end - z_start
    return z_start, z_end, slices

# def save_coarse_predictions(loc_dir, pred,valid_patlist,valid_sublist, z_VALID):
#     y = 0
#     z_starttrain = []
#     z_endtrain = [0]*len(valid_patlist)
#     roi_dict = {0:2,1:4,2:5,3:7}
#     for i in range(len(valid_patlist)):
#         # roi1 = np.zeros((512, 512, z_VALID[i]))
#         # roi2 = np.zeros((512, 512, z_VALID[i]))
#         # roi3 = np.zeros((512, 512, z_VALID[i]))
#         # roi4 = np.zeros((512, 512, z_VALID[i]))
#         roi5 = np.zeros((512, 512, z_VALID[i]))
#         # roi7 = np.zeros((512, 512, z_VALID[i]))
#         # roi1[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 0], (1, 2, 0))
#         # roi2[:, 0:256, :] = roi1[:, 0:256, :]
#         # roi3[:, 256:512, :] = roi1[:, 256:512, :]
#         # roi4[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 1], (1, 2, 0))
#         roi5[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 2], (1, 2, 0))
#         # roi7[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 3], (1, 2, 0))
#         z_start1check = 0
#         count1 = 0
#         for z1 in range(0, z_VALID[i]):
#             if np.sum(roi5[:, :, z1]) > 50 and count1 == 0:
#                 if z_start1check == 0:
#                     z_start1check = 1
#                     z_starttrain.append(z1)
#                     count1 = 1
#                 else:
#                     count1 = 1
#             if count1 == 1 and np.sum(roi5[:, :, z1]) < 50:
#                 if z1 > z_endtrain[i]:
#                     z_endtrain[i] = z1
#                     count1 = 0
#         slices = z_endtrain[i] - z_starttrain[i]
#
#         while slices < 10:
#             z_start1check = 0
#             for z1 in range(z_endtrain[i] + 1, z_VALID[i]):
#                 if (z_start1check == 0) and (np.sum(roi5[:, :, z1]) > 50):
#                     z_starttrain[i] = z1
#                     z_start1check = 1
#                 if z_start1check == 1 and (np.sum(roi5[:, :, z1]) < 50):
#                     z_endtrain[i] = z1
#                     z_start1check = 2
#             slices = z_endtrain[i] - z_starttrain[i]
#         roi5[:, :, 0:z_starttrain[i]] = 0
#         roi5[:, :, z_endtrain[i]:z_VALID[i]] = 0
#         # np.save(os.path.join(loc_dir, 'ROI_{}_{}_2_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi2)
#         # np.save(os.path.join(loc_dir, 'ROI_{}_{}_3_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi3)
#         # np.save(os.path.join(loc_dir, 'ROI_{}_{}_4_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi4)
#         np.save(os.path.join(loc_dir, 'ROI_{}_{}_5_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi5)
#         # np.save(os.path.join(loc_dir, 'ROI_{}_{}_7_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi7)
#         y += z_VALID[i]

def save_coarse_predictions(loc_dir, pred,valid_patlist,valid_sublist, z_VALID, roi_list=[0,1,2,3],roi_dict = {0: 2, 1: 4, 2: 5, 3: 7}):

    y = 0
    for i in range(len(valid_patlist)):
        if 0 in roi_list:
            print(roi_dict[0])
            roi = np.zeros((512, 512, z_VALID[i]))
            roi1 = np.zeros((512, 512, z_VALID[i]))
            roi2 = np.zeros((512, 512, z_VALID[i]))
            roi[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 0], (1, 2, 0))
            roi = clean_coarse_preds(roi, z_VALID[i])
            roi1[:, 0:256, :] = roi[:, 0:256, :]
            roi2[:, 256:512, :] = roi[:, 256:512, :]
            np.save(os.path.join(loc_dir,'ROI_{}_{}_{}_coarse.npy'.format(valid_patlist[i], valid_sublist[i], 2)),roi1)
            np.save(os.path.join(loc_dir,'ROI_{}_{}_{}_coarse.npy'.format(valid_patlist[i], valid_sublist[i], 3)),roi2)
            roi_list.pop(0)
        for r in roi_list:
            roi = np.zeros((512, 512, z_VALID[i]))
            roi[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, r], (1, 2, 0))
            roi = clean_coarse_preds(roi,z_VALID[i])
            np.save(os.path.join(loc_dir, 'ROI_{}_{}_{}_coarse.npy'.format(valid_patlist[i], valid_sublist[i], roi_dict[r])), roi)
        y += z_VALID[i]

def save_coarse_predictions_sch(loc_dir, pred,valid_patlist,valid_sublist, z_VALID, roi_list=[0],roi_dict = {0: 5}):

    y = 0
    for i in range(len(valid_patlist)):
        roi = np.zeros((512, 512, z_VALID[i]))
        roi[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 0], (1, 2, 0))
        roi = clean_coarse_preds(roi,z_VALID[i])
        np.save(os.path.join(loc_dir, 'ROI_{}_{}_{}_coarse.npy'.format(valid_patlist[i], valid_sublist[i], roi_dict[0])), roi)
        y += z_VALID[i]

def clean_coarse_preds(roi, z_VALID):
    z_endtrain = 0
    z_start1check = 0
    count1 = 0
    for z1 in range(0, z_VALID):
        if np.sum(roi[:, :, z1]) > 50 and count1 == 0:
            if z_start1check == 0:
                z_start1check = 1
                z_starttrain = z1
                count1 = 1
            else:
                count1 = 1
        if count1 == 1 and np.sum(roi[:, :, z1]) < 50:
            if z1 > z_endtrain:
                z_endtrain = z1
                count1 = 0
    slices = z_endtrain - z_starttrain

    while slices < 10:
        z_start1check = 0
        for z1 in range(z_endtrain + 1, z_VALID):
            if (z_start1check == 0) and (np.sum(roi[:, :, z1]) > 50):
                z_starttrain = z1
                z_start1check = 1
            if z_start1check == 1 and (np.sum(roi[:, :, z1]) < 50):
                z_endtrain = z1
                z_start1check = 2
        slices = z_endtrain- z_starttrain
    roi[:, :, 0:z_starttrain] = 0
    roi[:, :, z_endtrain:z_VALID] = 0
    return roi

def save_fine_predictions(loc_dir, pred,valid_patlist,valid_sublist, z_VALID, z_start, z_end, x_cent, y_cent, img_rows, img_cols, roi_interest, name):
    for i in range(len(valid_patlist)):
        roi = np.zeros((512, 512, z_VALID[i]))
        roi[x_cent[i]-int(img_rows/2): x_cent[i]+int(img_rows/2), y_cent[i]-int(img_cols/2): y_cent[i]+int(img_cols/2), z_start[i]:z_end[i]] = pred[i,:, :, :, 0]
        np.save(os.path.join(loc_dir, 'ROI_{}_{}_{}_{}.npy'.format(valid_patlist[i], valid_sublist[i],roi_interest,name)), roi)

def predict_with_uncertainty(f, x, n_iter=2, threshold=0.5):
    pred = np.zeros((n_iter,) + (x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1))
    for i in range(n_iter):
        pred[i] = f((x, 1))[0]
    Pred_Mean = np.mean(pred, axis=0) > threshold
    Pred = np.mean(pred, axis=0)
    Pred_var = np.std(pred, axis=0)
    pred_flat = Pred.flatten()
    pred_flat_up = Pred.flatten()
    pred_flat_down = Pred.flatten()
    var_flat = Pred_var.flatten()
    for j in range(len(pred_flat)):
        pred_flat_up[j] = pred_flat[j] + 2 * var_flat[j]
        pred_flat_down[j] = pred_flat[j] - 2 * var_flat[j]

    bounds_high = pred_flat_up.reshape((160, 160, 64)) > threshold
    bounds_low = pred_flat_down.reshape((160, 160, 64)) > threshold
    return Pred_Mean, bounds_high, bounds_low

def calculating_slicesofinterest_segtrain(train_patlist, train_subpatlist, z_TRAIN, roi_interest):
    z_starttrain = []  # [0]*len(TRAIN)
    z_endtrain = [0] * len(train_patlist)
    i = 0
    roi = [roi_interest]
    for x1 in train_patlist:
        j1 = train_subpatlist[i]
        z_start1check = 0
        for nroi in roi:
            count1 = 0
            roi_file = np.load(os.path.join(data_dir, 'ROI_{0}_{1}_{2}.npy'.format(x1, j1, nroi)))
            for z1 in range(0, z_TRAIN[i]):
                SUM = np.sum(roi_file[:, :, z1])
                if SUM > 0 and count1 == 0:
                    if z_start1check == 0:
                        z_start1check = 1
                        z_starttrain.append(z1)
                        count1 = 1
                    else:
                        count1 = 1
                if count1 == 1 and SUM < 1:
                    if z1 > z_endtrain[i]:
                        z_endtrain[i] = z1
                        count1 = 0
        i = i + 1
    return z_starttrain, z_endtrain

def calculating_slicesofinterest_segvalid(loc_dir,patlist, sublist, z_dim, nroi):
    z_starttrain = []
    z_endtrain = [0] * len(patlist)
    i = 0
    for x1 in patlist:
        j1 = sublist[i]
        z_start1check = 0
        count1 = 0
        roi_file = np.load(os.path.join(loc_dir, 'ROI_{0}_{1}_{2}_coarse.npy'.format(x1, j1, nroi)))
        for z1 in range(0, z_dim[i]):
            SUM = np.sum(roi_file[:, :, z1])
            if SUM > 50 and count1 == 0:
                if z_start1check == 0:
                    z_start1check = 1
                    z_starttrain.append(z1)
                    count1 = 1
                else:
                    count1 = 1
            if count1 == 1 and SUM < 50:
                if z1 > z_endtrain[i]:
                    z_endtrain[i] = z1
                    count1 = 0
        i = i + 1
    return z_starttrain, z_endtrain

def calculating_slicesofinterestfor3D(train_patlist, z_starttrain, z_endtrain, z_TRAIN, buff_size=32):
    z_starttrainact = []
    z_endtrainact = []
    for l in range(0, len(train_patlist)):
        slices = z_endtrain[l] - z_starttrain[l]
        additional = buff_size - slices
        leftadd = int(additional/2)
        rightadd = additional - leftadd
        if (z_starttrain[l] >= leftadd) and (z_TRAIN[l] - z_endtrain[l] >= rightadd):
            z_starttrainact.append(z_starttrain[l] - leftadd)
            z_endtrainact.append(z_endtrain[l] + rightadd)
        elif (z_starttrain[l] < leftadd) and (z_TRAIN[l] - z_endtrain[l] > rightadd):
            z_starttrainact.append(0)
            z_endtrainact.append(buff_size)
        elif (z_starttrain[l] > leftadd) and (z_TRAIN[l] - z_endtrain[l] < rightadd):
            z_starttrainact.append(z_TRAIN[l] - buff_size)
            z_endtrainact.append(z_TRAIN[l])
        elif (z_starttrain[l] < leftadd) and (z_TRAIN[l] - z_endtrain[l] < rightadd):
            z_starttrainact.append(0)
            z_endtrainact.append(z_TRAIN[l])
    return z_starttrainact, z_endtrainact

def calculate_centroid_segtrain(train_patlist, train_subpatlist, z_starttrainact, z_endtrainact, nroi):
    sl1 = 0
    x_centtrain = []
    y_centtrain = []
    for x1 in train_patlist:
        t1 = train_subpatlist[sl1]
        y_cent = 0
        x_cent = 0
        count = 0
        a = []
        roi_file = np.load(os.path.join(data_dir, 'ROI_{0}_{1}_{2}.npy'.format(x1, t1, nroi)))
        for z1 in range(z_starttrainact[sl1], z_endtrainact[sl1]):
            img_roi = roi_file[:,:,z1]
            label_img = label(img_roi)
            regions = regionprops(label_img)
            for props in regions:
                x0, y0 = props.centroid
                count += 1
                a.append(y0)
                y_cent += y0
                x_cent += x0
        y = y_cent / count
        x = x_cent / count
        x_centtrain.append(int(x))
        y_centtrain.append(int(y))
        sl1 += 1
    return x_centtrain, y_centtrain

def calculate_centroid_segvalid(loc_dir, train_patlist, train_subpatlist, z_starttrainact, z_endtrainact, nroi):
    sl1 = 0
    x_centtrain = []
    y_centtrain = []
    for x1 in train_patlist:
        t1 = train_subpatlist[sl1]
        y_cent = 0
        x_cent = 0
        count = 0
        a = []
        roi_file = np.load(os.path.join(loc_dir, 'ROI_{0}_{1}_{2}_coarse.npy'.format(x1, t1, nroi)))
        for z1 in range(z_starttrainact[sl1], z_endtrainact[sl1]):
            img_roi = roi_file[:,:,z1]
            label_img = label(img_roi)
            regions = regionprops(label_img)
            for props in regions:
                x0, y0 = props.centroid
                count += 1
                a.append(y0)
                y_cent += y0
                x_cent += x0
        y = y_cent / count
        x = x_cent / count
        x_centtrain.append(int(x))
        y_centtrain.append(int(y))
        sl1 += 1
    return x_centtrain, y_centtrain

def image_histogram_equalization(image, number_bins=128):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

