from __future__ import print_function       # use print function of Python3 in Python2

import os                                   # handle files and directories
import datetime                             # give time
import random                               # used as shuffle
import json                                 # output json file
import argparse                             # resolve input parameters
import densenet                             # import dense neural network
import numpy as np                          # handle arrays
import keras.backend as K                   # use tensorflow

from keras.optimizers import Adam           # use optimizers

import data_loader                          # self-defined file to load images

def run_MURA(
                batch_size=8,                   # select a batch of samples to train a time
                nb_epoch=12,                    # times of iteration
                depth=22,                       # network depth
                nb_dense_block=4,               # number of dense blocks
                nb_filter=16,                   # initial number of conv filter
                growth_rate=12,                 # numbers of new filters added by each layer
                dropout_rate=0.2,               # dropout rate
                learning_rate=0.001,            # learning rate
                weight_decay=1E-4,              # wight decay
                plot_architecture=False         # plot network architecture
):

    ###################
    # Data processing #
    ###################

    im_size = 320   # resize images
    path_train = '/home/yu/Documents/tensorflow/MURA/MURA-v1.1/train/XR_ELBOW'      # the absolute path
    path_valid = '/home/yu/Documents/tensorflow/MURA/MURA-v1.1/valid/XR_ELBOW'
    X_train_path,Y_train = data_loader.load_path(root_path = path_train, size = im_size)
    X_valid_path,Y_valid = data_loader.load_path(root_path = path_valid, size = im_size)
    
    X_valid = data_loader.load_image(X_valid_path, im_size)     # import path for validation
    Y_valid = np.asarray(Y_valid)
    nb_classes = 1                                
    img_dim = (im_size, im_size, 1)     #tuple channel last

    
    ###################
    # Construct model #
    ###################

    # model is one instance of class 'Model'
    model = densenet.DenseNet(nb_classes,
                              img_dim,
                              depth,
                              nb_dense_block,
                              growth_rate,
                              nb_filter,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    # Model output
    model.summary()

    # Build optimizer
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,                # optimizer used to update gradient
                  metrics=["accuracy"])

    if plot_architecture:
        from keras.utils import plot_model
        plot_model(model, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################

    print("Start Training")

    list_train_loss = []
    list_valid_loss = []
    list_learning_rate = []
    best_record = [100,0,100,100]     # record the best result
    start_time = datetime.datetime.now()
    for e in range(nb_epoch):

        if e == int(0.25 * nb_epoch):     # update learning_rate
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

        if e == int(0.5 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 50.))

        if e == int(0.75 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

        split_size = batch_size
        num_splits = len(X_train_path) / split_size     # Calculate how many batches of training images
        arr_all = np.arange(len(X_train_path)).astype(int)      # Return evenly spaced values within a given interval
        random.shuffle(arr_all)     # reshuffle, so the order of each training would be different
                                    # avoid local optimal solution
                                    # with shuffle open, it would be SGD
        arr_splits = np.array_split(arr_all, num_splits)    # Divede the training images to num_splits batches

        l_train_loss = []
        batch_train_loss = []
        start = datetime.datetime.now()

        for i, batch_idx in enumerate(arr_splits):      # i: how many batches, batch_idx: each batch

            X_batch_path, Y_batch = [], []              # X_batch_path is the path of images, Y_batch is the label

            for idx in batch_idx:

                X_batch_path.append(X_train_path[idx])
                Y_batch.append(Y_train[idx])

            X_batch = data_loader.load_image(Path = X_batch_path, size =im_size)      # load data for training
            Y_batch = np.asarray(Y_batch)       # Transform the type of Y_batch as array, that is label
            train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)     # train, return loss and accuracy

            l_train_loss.append([train_logloss, train_acc])
            batch_train_loss.append([train_logloss, train_acc])
            if i %100 == 0:                     # 100 batches
                loss_1, acc_1 = np.mean(np.array(l_train_loss), 0)
                loss_2, acc_2 = np.mean(np.array(batch_train_loss), 0)
                batch_train_loss = []           
                print ('[Epoch {}/{}] [Batch {}/{}] [Time: {}] [all_batchs--> train_epoch_logloss: {:.5f}, train_epoch_acc:{:.5f}] '.format
                    (e+1,nb_epoch,i, len(arr_splits),datetime.datetime.now() - start,loss_1,acc_1),
                    '[this_100_batchs-->train_batchs_logloss: {:.5f}, train_batchs_acc:{:.5f}]'.format(loss_2, acc_2))

        # validate
        valid_logloss, valid_acc = model.evaluate(X_valid,
                                                Y_valid,
                                                verbose=0,
                                                batch_size=64)

        list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
        list_valid_loss.append([valid_logloss, valid_acc])
        list_learning_rate.append(float(K.get_value(model.optimizer.lr)))

        # to convert numpy array to json serializable
        print('[Epoch %s/%s] [Time: %s, Total_time: %s]' % (e + 1, nb_epoch, datetime.datetime.now() - start,
            datetime.datetime.now() - start_time),end = '')
        print('[train_loss_and_acc:{:.5f} {:.5f}] [valid_loss_acc:{:.5f} {:.5f}]'.format(list_train_loss[-1][0],
            list_train_loss[-1][1],list_valid_loss[-1][0],list_valid_loss[-1][1]))


        d_log = {}
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        d_log["optimizer"] = opt.get_config()
        d_log["train_loss"] = list_train_loss
        d_log["valid_loss"] = list_valid_loss
        d_log["learning_rate"] = list_learning_rate

        json_file = os.path.join('./log/experiment_log_MURA.json')

        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)

        record = [valid_logloss,valid_acc,abs(valid_logloss-list_train_loss[-1][0]),abs(valid_acc-list_train_loss[-1][1]),]
        if ((record[0]<=best_record[0]) &(record[1]>=best_record[1])) :
            if e <= int(0.25 * nb_epoch)|(record[2]<=best_record[2])&(record[3]<=best_record[3]):
                best_record=record                      
                print('saving the best model:epoch',e+1,best_record)
                model.save('save_models/best_MURA_modle@epochs{}.h5'.format(e+1))
        model.save('save_models/MURA_modle@epochs{}.h5'.format(e+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MURA experiment')
    parser.add_argument('--batch_size', default=8, type=int, #default=64
                        help='Batch size')
    parser.add_argument('--nb_epoch',  type=int, default=1,#default=30,
                        help='Number of epochs')
    parser.add_argument('--depth', type=int, default=6*3+4,#default=7,    
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=4, #default=1,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3, #default=1E-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4,
                        help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=True,#default=False,
                        help='Save a plot of the network architecture')

    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    list_dir = ["./log", "./figures", "./save_models"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    run_MURA(args.batch_size,
             args.nb_epoch,
             args.depth,
             args.nb_dense_block,
             args.nb_filter,
             args.growth_rate,
             args.dropout_rate,
             args.learning_rate,
             args.weight_decay,
             args.plot_architecture)

