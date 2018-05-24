

import tensorflow as tf
import cv2
import os
import numpy as np

from utils import load_data, sample_train_data, image_scaling, image_scaling_inverse
from model import CycleGAN

def train(img_A_dir, img_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir):

    np.random.seed(random_seed)

    num_epochs = 200
    mini_batch_size = 1
    learning_rate = 0.0002
    input_size = [256, 256, 3]
    num_filters = 64

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        if not os.path.exists(validation_B_output_dir):
            os.makedirs(validation_B_output_dir)

    model = CycleGAN(input_size = input_size, num_filters = num_filters)

    dataset_A_raw = load_data(img_dir = img_A_dir, load_size = 256)
    dataset_B_raw = load_data(img_dir = img_B_dir, load_size = 256)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)

        dataset_A, dataset_B = sample_train_data(dataset_A_raw, dataset_B_raw, load_size = 286, output_size = 256)

        n_samples = dataset_A.shape[0]
        for i in range(n_samples // mini_batch_size):
            print('Minibatch: %d' % i)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss = model.train(input_A = dataset_A[start:end], input_B = dataset_B[start:end], learning_rate = learning_rate)

            print('Generator Loss : %f' % generator_loss)
            print('Discriminator Loss : %f' % discriminator_loss)

        model.save(directory = model_dir, filename = model_name)

        for file in os.listdir(validation_A_dir):
            filepath = os.path.join(validation_A_dir, file)
            img = cv2.imread(filepath)
            img = image_scaling(imgs = img)
            img_converted = model.test(inputs = np.array([img]), direction = 'A2B')[0]
            img_converted = image_scaling_inverse(imgs = img_converted)
            cv2.imwrite(os.path.join(validation_A_output_dir, os.path.basename(file)), img_converted)

        for file in os.listdir(validation_B_dir):
            filepath = os.path.join(validation_B_dir, file)
            img = cv2.imread(filepath)
            img = image_scaling(imgs = img)
            img_converted = model.test(inputs = np.array([img]), direction = 'B2A')[0]
            img_converted = image_scaling_inverse(imgs = img_converted)
            cv2.imwrite(os.path.join(validation_B_output_dir, os.path.basename(file)), img_converted)

if __name__ == '__main__':

    img_A_dir = './data/horse2zebra/trainA'
    img_B_dir = './data/horse2zebra/trainB'
    model_dir = './model'
    model_name = 'horse_zebra.ckpt'
    random_seed = 0
    validation_A_dir = './data/horse2zebra/testA'
    validation_B_dir = './data/horse2zebra/testB'
    output_dir = './validation_output'

    train(img_A_dir = img_A_dir, img_B_dir = img_B_dir, model_dir = model_dir, model_name = model_name, random_seed = random_seed, validation_A_dir = validation_A_dir, validation_B_dir = validation_B_dir, output_dir = output_dir)