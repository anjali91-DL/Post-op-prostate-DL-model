import os
import numpy as np
from skimage.measure import label, regionprops


data_dir = '/data/mnt/share/dan/edrive/Anjali_Backup/BED_OAR_MASKS'

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

def save_coarse_predictions(loc_dir, pred):
    valid_patlist = np.load(os.path.join(loc_dir, 'valid.npy'))
    valid_sublist = np.load(os.path.join(loc_dir, 'valid_sub.npy'))
    z_VALID = np.load(os.path.join(loc_dir, 'z_valid.npy'))
    y = 0
    for i in range(len(valid_patlist)):
        roi1 = np.zeros((512, 512, z_VALID[i]))
        roi2 = np.zeros((512, 512, z_VALID[i]))
        roi3 = np.zeros((512, 512, z_VALID[i]))
        roi4 = np.zeros((512, 512, z_VALID[i]))
        roi5 = np.zeros((512, 512, z_VALID[i]))
        roi7 = np.zeros((512, 512, z_VALID[i]))

        print(i, pred[y:y + z_VALID[i], :, 0:160, 0].shape)
        print(i, pred[y:y + z_VALID[i], :, 160:320, 0].shape)
        roi1[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 0], (1, 2, 0))
        roi2[:, 0:256, :] = roi1[:, 0:256, :]
        roi3[:, 256:512, :] = roi1[:, 256:512, :]
        roi4[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 1], (1, 2, 0))
        roi5[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 2], (1, 2, 0))
        roi7[124:380, 100:420, :] = np.transpose(pred[y:y + z_VALID[i], :, :, 3], (1, 2, 0))

        np.save(os.path.join(loc_dir, 'ROI_{}_{}_2_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi2)
        np.save(os.path.join(loc_dir, 'ROI_{}_{}_3_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi3)
        np.save(os.path.join(loc_dir, 'ROI_{}_{}_4_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi4)
        np.save(os.path.join(loc_dir, 'ROI_{}_{}_5_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi5)
        np.save(os.path.join(loc_dir, 'ROI_{}_{}_7_coarse.npy'.format(valid_patlist[i], valid_sublist[i])), roi7)

        y += z_VALID[i]

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

def calculating_slicesofinterest_segvalid(patlist, sublist, z_dim, nroi):
    z_starttrain = []
    z_endtrain = [0] * len(patlist)
    i = 0
    for x1 in patlist:
        j1 = sublist[i]
        z_start1check = 0
        count1 = 0
        roi_file = np.load(os.path.join(data_dir, 'ROI_{0}_{1}_{2}_coarse.npy'.format(x1, j1, nroi)))
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
        slices = z_endtrain[i] - z_starttrain[i]

        while slices < 10:
            z_start1check = 0
            for z1 in range(z_endtrain[i] + 1, z_dim[i]):
                if (z_start1check == 0) and (np.sum(roi_file[:, :, z1]) > 50):
                    z_starttrain[i] = z1
                    z_start1check = 1
                if z_start1check == 1 and (np.sum(roi_file[:, :, z1]) < 50):
                    z_endtrain[i] = z1
                    z_start1check = 2
            slices = z_endtrain[i] - z_starttrain[i]
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

def calculate_centroid_segvalid(train_patlist, train_subpatlist, z_starttrainact, z_endtrainact, nroi):
    sl1 = 0
    x_centtrain = []
    y_centtrain = []
    for x1 in train_patlist:
        t1 = train_subpatlist[sl1]
        y_cent = 0
        x_cent = 0
        count = 0
        a = []
        roi_file = np.load(os.path.join(data_dir, 'ROI_{0}_{1}_{2}_coarse.npy'.format(x1, t1, nroi)))
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