import os
import datetime
import json
from collections import OrderedDict                     # want to get the json file in order, not succeed

from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau


def make_vdirs_for_keras(dir_list, catdir_name):        # divide all data to be positive and negative
                                                        # the class for MURA should be two
                                                        # preprocess data to adjust the format
    os.makedirs(catdir_name, exist_ok=True)             # create subdirectory
    dir_pos = os.path.join(catdir_name,'positive')
    dir_neg = os.path.join(catdir_name,'negative')

    os.makedirs(dir_pos, exist_ok=True)                 # create subdirectory ./v_train/XR_WRIST/positive
    os.makedirs(dir_neg, exist_ok=True)                 # create subdirectory ./v_train_XR_WRIST/negative
    sym_dirs = []                                       # return figure and label

    for root, dirs, files in os.walk(dir_list):
        if files:
            base_root = root.split('/')[9]
            if root.split('_')[-1] == 'positive' :
                n_dir = os.path.join(dir_pos, base_root)  # create the absolute path
            else:
                n_dir = os.path.join(dir_neg, base_root)  # create the absolute path
            # print('n_dir', n_dir)
            if not os.path.isfile(n_dir) :                # judge whether the file exist already
                try:
                    os.symlink(os.path.abspath(root),
                               n_dir)
                                           
                except FileExistsError:                   # raise error
                    os.remove(n_dir)
                    os.symlink(os.path.abspath(root),
                               n_dir)

            sym_dirs += [n_dir]
#    return sym_dirs


def train_mobilenet(image_class, epochs):

    start_train_time = datetime.datetime.now()
    image_class = image_class
    root_path = '/home/yu/Documents/tensorflow/MURA/MURA-v1.1/'                     # the root path of dataset
    train_dirs = os.path.join(root_path, 'train/{}'.format(image_class))            # import data for training
    valid_dirs = os.path.join(root_path, 'valid/{}'.format(image_class))            # import data for validation

    if not os.path.exists('v_train/{}'.format(image_class)):                        # iterate to create symbolic link to data
        make_vdirs_for_keras(train_dirs, 'v_train/{}'.format(image_class))

    if not os.path.exists('v_valid/{}'.format(image_class)):        
        make_vdirs_for_keras(valid_dirs, 'v_valid/{}'.format(image_class))

    idg_train_settings = dict(
                            samplewise_center = True,
                            samplewise_std_normalization = True,
                            rotation_range = 5,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1,
                            zoom_range = 0.1,
                            horizontal_flip = True,
                            vertical_flip = True)
    idg_train = ImageDataGenerator(**idg_train_settings)

    idg_valid_settings = dict(
                            samplewise_center = True,
                            samplewise_std_normalization = True,
                            rotation_range = 0,
                            width_shift_range = 0.,
                            height_shift_range = 0.,
                            zoom_range = 0.0,
                            horizontal_flip = False,
                            vertical_flip = False)
    idg_valid = ImageDataGenerator(**idg_valid_settings)

    train_gen = idg_train.flow_from_directory(
                                            'v_train/{}'.format(image_class),
                                            follow_links = True,
                                            target_size=(128, 128),
                                            color_mode = 'grayscale')

    valid_gen = idg_valid.flow_from_directory(
                                           'v_valid/{}'.format(image_class),
                                            follow_links = True,
                                            target_size=(128, 128),
                                            color_mode = 'grayscale')

    a, b = next(valid_gen)
    s_net = MobileNet(classes=b.shape[1], weights = None, input_shape=a.shape[1:])
    # s_net.summary()

    s_net.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])
    # print('Layers: {}, parameters: {}'.format(len(s_net.layers), s_net.count_params()))

    if not os.path.exists('weights'):
        os.mkdir('weights/')
    file_path="weights/weights.best.hdf5."+image_class
    checkpoint = ModelCheckpoint(
                                file_path,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min')
    early = EarlyStopping(monitor="val_acc",
                          mode="max",
                          patience=3)
    callbacks_list = [checkpoint, early]            #early

    s_net.fit_generator(
                    train_gen,
                    steps_per_epoch=30,             # default 30
                    validation_data=valid_gen,
                    validation_steps=10,
                    epochs=epochs,
                    callbacks=callbacks_list)

    end_train_time = datetime.datetime.now()
    time_train = end_train_time - start_train_time
    return time_train, s_net, valid_gen
#    print('Total training time: %s' % (end_train_time - start_time))


def evaluate_mobilenet(model_evaluate, valid_gen):
    model_evaluate = model_evaluate
    loss_evaluate, acc_evaluate = model_evaluate.evaluate_generator(valid_gen, steps=10, verbose=1)
    return loss_evaluate, acc_evaluate


def log_mobilenet(data, time, loss, acc):
    if not os.path.exists('log'):
        os.mkdir('log')
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')+'.json'
    json_file = os.path.join('./log', mkfile_time)                          # use time to name the log file
    with open(json_file, 'a') as fp:
        for i, image in enumerate(data):
            d_log = {}
            d_log['Class'] = image
            d_log['Time for training'] = time[i]
            d_log['Loss_evaluate'] = loss[i]
            d_log['Accuracy_evaluate'] = acc[i]
            # json.dump(d_log, fp, indent=4, sort_keys=True, default=str)
            json.dump(OrderedDict(d_log), fp, indent=4, default=str)

if __name__=='__main__':
    data = ['XR_ELBOW',
            'XR_FINGER',
            'XR_FOREARM',
            'XR_HAND',
            'XR_HUMERUS',
            'XR_SHOULDER',
            'XR_WRIST']
    # data = ['XR_WRIST', 'XR_HAND']
    # data = ['XR_WRIST']
    time_all = []
    loss_all = []
    acc_all = []
    for image in data:
        time, model, valid_gen = train_mobilenet(image, 30)
        [loss, acc] = evaluate_mobilenet(model, valid_gen)
        time_all.append(time)
        loss_all.append(loss)
        acc_all.append(acc)
    for i, image in enumerate(data):
        print('Model for %s' % image)
        print('Total training time: %s' % time_all[i])
        print('loss: %s, accuracy: %s' % (loss_all[i], acc_all[i]))
    log_mobilenet(data, time_all, loss_all, acc_all)

