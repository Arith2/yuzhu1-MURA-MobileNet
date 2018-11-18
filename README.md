# yuzhu1-MURA-MobileNet
Keras, MURA, MobileNet, 

Topic:Fractures detection on X-Ray images using Machine Learning

Purpose: To implement a machine learning model in Raspberry Pi

Configuration:
Ubuntu 16.04,
Tensorflow,
Virtualenv,
Keras,
Python 3.5.

11-17-2018
1）Run a current project based on DNN successfully, https://github.com/JarvisPeng/MURA_densenet.git.
2) Log file in json format:
{
    "batch_size": 8,
    "learning_rate": [
        9.999999747378752e-06
    ],
    "nb_epoch": 1,
    "optimizer": {
        "amsgrad": false,
        "beta_1": 0.8999999761581421,
        "beta_2": 0.9990000128746033,
        "decay": 0.0,
        "epsilon": 1e-08,
        "lr": 9.999999747378752e-06
    },
    "train_loss": [
        [
            5.6305108070373535,
            0.5964781641960144
        ]
    ],
    "valid_loss": [
        [
            5.66788777382143,
            0.5075268826177043
        ]
    ]
}
3) Next step: use MobileNet to replace DNN to reduce the size of network and the consumption of resources
