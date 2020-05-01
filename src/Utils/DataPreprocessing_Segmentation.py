import os
import numpy as np
from src.Utils.Data_Utils import z_calc,calculating_slicesofinterest_segtrain, calculating_slicesofinterest_segvalid, \
    calculating_slicesofinterestfor3D, calculate_centroid_segtrain, calculate_centroid_segvalid

def preprocessing_train(main_pred_dir,train_pat, train_pat_sublist, z_train, roi_interest, img_slcs, justload = 1):
    if justload:
        z_starttrain = np.load(os.path.join(main_pred_dir, 'z_starttrain.npy'))
        z_endtrain = np.load(os.path.join(main_pred_dir, 'z_endtrain.npy'))
        x_centtrain = np.load(os.path.join(main_pred_dir, 'x_centtrain.npy'))
        y_centtrain = np.load(os.path.join(main_pred_dir, 'y_centtrain.npy'))
    else:
        z_starttrain, z_endtrain = calculating_slicesofinterest_segtrain(train_pat, train_pat_sublist, z_train, roi_interest)
        z_starttrain, z_endtrain = calculating_slicesofinterestfor3D(train_pat,z_starttrain, z_endtrain, z_train, img_slcs)
        x_centtrain, y_centtrain = calculate_centroid_segtrain(train_pat,
           train_pat_sublist, z_starttrain, z_endtrain, roi_interest)
        np.save(os.path.join(main_pred_dir, 'train.npy'),train_pat)
        np.save(os.path.join(main_pred_dir, 'train_sub.npy'), train_pat_sublist)
        np.save(os.path.join(main_pred_dir, 'z_train.npy'), z_train)
        np.save(os.path.join(main_pred_dir, 'z_starttrain.npy'), z_starttrain)
        np.save(os.path.join(main_pred_dir, 'z_endtrain.npy'), z_endtrain)
        np.save(os.path.join(main_pred_dir, 'x_centtrain.npy'), x_centtrain)
        np.save(os.path.join(main_pred_dir, 'y_centtrain.npy'), y_centtrain)
    return z_starttrain, z_endtrain, x_centtrain, y_centtrain

def preprocessing_valid(main_pred_dir,valid_pat, valid_sub_pat, z_valid, roi_interest, img_slcs, justload = 1):
    if justload:
        z_startvalid = np.load(os.path.join(main_pred_dir, 'z_startvalid.npy'))
        z_endvalid = np.load(os.path.join(main_pred_dir, 'z_endvalid.npy'))
        x_centvalid = np.load(os.path.join(main_pred_dir, 'x_centvalid.npy'))
        y_centvalid = np.load(os.path.join(main_pred_dir, 'y_centvalid.npy'))
    else:
        z_startvalid, z_endvalid = calculating_slicesofinterest_segtrain(valid_pat, valid_sub_pat, z_valid, roi_interest)
        z_startvalid, z_endvalid = calculating_slicesofinterestfor3D(valid_pat, z_startvalid, z_endvalid, z_valid, img_slcs)
        x_centvalid, y_centvalid = calculate_centroid_segtrain(valid_pat, valid_sub_pat, z_startvalid, z_endvalid, roi_interest)
        np.save(os.path.join(main_pred_dir, 'valid.npy'), valid_pat)
        np.save(os.path.join(main_pred_dir, 'valid_sub.npy'), valid_sub_pat)
        np.save(os.path.join(main_pred_dir, 'z_valid.npy'), z_valid)
        np.save(os.path.join(main_pred_dir, 'z_startvalid.npy'), z_startvalid)
        np.save(os.path.join(main_pred_dir, 'z_endvalid.npy'), z_endvalid)
        np.save(os.path.join(main_pred_dir, 'x_centvalid.npy'), x_centvalid)
        np.save(os.path.join(main_pred_dir, 'y_centvalid.npy'), y_centvalid)
    return z_startvalid, z_endvalid, x_centvalid, y_centvalid

def preprocessing_test(main_pred_dir,test_pat, test_sub_pat, z_test, roi_interest, img_slcs, justload = 1):
    if justload:
        z_starttest = np.load(os.path.join(main_pred_dir, 'z_starttest.npy'))
        z_endtest = np.load(os.path.join(main_pred_dir, 'z_endtest.npy'))
        x_centtest = np.load(os.path.join(main_pred_dir, 'x_centtest.npy'))
        y_centtest = np.load(os.path.join(main_pred_dir, 'y_centtest.npy'))
    else:
        z_starttest, z_endtest = calculating_slicesofinterest_segvalid(main_pred_dir,test_pat, test_sub_pat, z_test, roi_interest)
        z_starttest, z_endtest = calculating_slicesofinterestfor3D(test_pat, z_starttest, z_endtest, z_test, img_slcs)
        x_centtest, y_centtest = calculate_centroid_segvalid(main_pred_dir,test_pat, test_sub_pat, z_starttest, z_endtest, roi_interest)
        np.save(os.path.join(main_pred_dir, 'z_test.npy'), z_test)
        np.save(os.path.join(main_pred_dir, 'z_starttest.npy'), z_starttest)
        np.save(os.path.join(main_pred_dir, 'z_endtest.npy'), z_endtest)
        np.save(os.path.join(main_pred_dir, 'x_centtest.npy'), x_centtest)
        np.save(os.path.join(main_pred_dir, 'y_centtest.npy'), y_centtest)
    return z_starttest, z_endtest, x_centtest, y_centtest

