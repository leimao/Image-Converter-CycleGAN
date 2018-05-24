

import tensorflow as tf

from utils import load_train_data, image_scaling_inverse
from model import CycleGAN
import cv2
import os
import numpy as np

def train(img_A_dir = './data/horse2zebra/trainA', img_B_dir = './data/horse2zebra/trainB'):

    num_epochs = 100
    mini_batch_size = 10
    #learning_rate = 0.002
    #learning_rate = 0.005
    learning_rate = 0.0002

    model = CycleGAN(input_size = [256, 256, 3], num_filters = 32)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)
        dataset_A, dataset_B = load_train_data(img_A_dir = img_A_dir, img_B_dir = img_B_dir)

        #dataset_A = dataset_A[0:20]
        #dataset_B = dataset_B[0:20]

        n_samples = dataset_A.shape[0]
        for i in range(n_samples // mini_batch_size):
            print('Minibatch: %d' % i)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss = model.train(input_A = dataset_A[start:end], input_B = dataset_B[start:end], learning_rate = learning_rate)

            print('Generator Loss : %f' % generator_loss)
            print('Discriminator Loss : %f' % discriminator_loss)

        model.save(directory = './model', filename = 'cyclegan.ckpt')

        testA_dir = './data/horse2zebra/testA'
        testB_dir = './data/horse2zebra/testB'
        demo_dir = './demo'
        demo_A_dir = os.path.join(demo_dir, 'convertedA')
        demo_B_dir = os.path.join(demo_dir, 'convertedB')

        if not os.path.exists(demo_dir):
            os.makedirs(demo_dir)
        if not os.path.exists(demo_A_dir):
            os.makedirs(demo_A_dir)
        if not os.path.exists(demo_B_dir):
            os.makedirs(demo_B_dir)

        #test_A_paths = [os.path.join(testA_dir, file) for file in os.listdir(testA_dir) if os.path.isfile(os.path.join(testA_dir, file))]
        #test_B_paths = [os.path.join(testB_dir, file) for file in os.listdir(testB_dir) if os.path.isfile(os.path.join(testB_dir, file))]

        for file in os.listdir(testA_dir):
            filepath = os.path.join(testA_dir, file)
            img = cv2.imread(filepath)
            conversion = model.test(inputs = np.array([img]), direction = 'A2B')[0]
            conversion = image_scaling_inverse(imgs = conversion)
            cv2.imwrite(os.path.join(demo_A_dir, os.path.basename(file)), conversion)

        for file in os.listdir(testB_dir):
            filepath = os.path.join(testB_dir, file)
            img = cv2.imread(filepath)
            conversion = model.test(inputs = np.array([img]), direction = 'B2A')[0]
            conversion = image_scaling_inverse(imgs = conversion)
            cv2.imwrite(os.path.join(demo_B_dir, os.path.basename(file)), conversion)

if __name__ == '__main__':

    train()