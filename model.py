
import os
import tensorflow as tf
from module import discriminator, generator_resnet
from utils import l1_loss, l2_loss, cross_entropy_loss



class CycleGAN(object):

    def __init__(self, input_size, num_filters = 64, discriminator = discriminator, generator = generator_resnet, lambda_cycle = 10):

        self.input_size = input_size

        self.discriminator = discriminator
        self.generator = generator
        self.lambda_cycle = lambda_cycle
        self.num_filters = num_filters

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):

        # Placeholders for real training samples
        self.input_A_real = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A_real')
        self.input_B_real = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B_real')
        # Placeholders for fake generated samples
        self.input_A_fake = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A_fake')
        self.input_B_fake = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B_fake')
        # Placeholder for test samples
        self.input_A_test = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A_test')
        self.input_B_test = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B_test')

        self.generation_B = self.generator(inputs = self.input_A_real, num_filters = self.num_filters, reuse = False, scope_name = 'generator_A2B')
        self.cycle_A = self.generator(inputs = self.generation_B, num_filters = self.num_filters, reuse = False, scope_name = 'generator_B2A')

        self.generation_A = self.generator(inputs = self.input_B_real, num_filters = self.num_filters, reuse = True, scope_name = 'generator_B2A')
        self.cycle_B = self.generator(inputs = self.generation_A, num_filters = self.num_filters, reuse = True, scope_name = 'generator_A2B')

        self.discrimination_A_fake = self.discriminator(inputs = self.generation_A, num_filters = self.num_filters, reuse = False, scope_name = 'discriminator_A')
        self.discrimination_B_fake = self.discriminator(inputs = self.generation_B, num_filters = self.num_filters, reuse = False, scope_name = 'discriminator_B')

        # Cycle loss
        self.cycle_loss = l1_loss(y = self.input_A_real, y_hat = self.cycle_A) + l1_loss(y = self.input_B_real, y_hat = self.cycle_B)

        # Generator loss
        # Generator wants to fool discriminator
        self.generator_loss_A2B = l2_loss(y = tf.ones_like(self.discrimination_B_fake), y_hat = self.discrimination_B_fake)
        self.generator_loss_B2A = l2_loss(y = tf.ones_like(self.discrimination_A_fake), y_hat = self.discrimination_A_fake)

        # Merge the two generators and the cycle loss
        self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + self.lambda_cycle * self.cycle_loss

        # Discriminator loss
        self.discrimination_input_A_real = self.discriminator(inputs = self.input_A_real, num_filters = self.num_filters, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_input_B_real = self.discriminator(inputs = self.input_B_real, num_filters = self.num_filters, reuse = True, scope_name = 'discriminator_B')
        self.discrimination_input_A_fake = self.discriminator(inputs = self.input_A_fake, num_filters = self.num_filters, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_input_B_fake = self.discriminator(inputs = self.input_B_fake, num_filters = self.num_filters, reuse = True, scope_name = 'discriminator_B')

        # Discriminator wants to classify real and fake correctly
        self.discriminator_loss_input_A_real = l2_loss(y = tf.ones_like(self.discrimination_input_A_real), y_hat = self.discrimination_input_A_real)
        self.discriminator_loss_input_A_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_A_fake), y_hat = self.discrimination_input_A_fake)
        self.discriminator_loss_A = (self.discriminator_loss_input_A_real + self.discriminator_loss_input_A_fake) / 2

        self.discriminator_loss_input_B_real = l2_loss(y = tf.ones_like(self.discrimination_input_B_real), y_hat = self.discrimination_input_B_real)
        self.discriminator_loss_input_B_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_B_fake), y_hat = self.discrimination_input_B_fake)
        self.discriminator_loss_B = (self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake) / 2

        # Merge the two discriminators into one
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        #print('===============================')
        #for var in t_vars: print(var.name)
        #print('===============================')
        #for var in self.d_vars: print(var.name)
        #print('===============================')
        #for var in self.g_vars: print(var.name)

        # Reserved for test
        self.generation_B_test = self.generator(inputs = self.input_A_test, num_filters = self.num_filters, reuse = True, scope_name = 'generator_A2B')
        self.generation_A_test = self.generator(inputs = self.input_B_test, num_filters = self.num_filters, reuse = True, scope_name = 'generator_B2A')


    def optimizer_initializer(self):

        self.learning_rate = tf.placeholder(tf.float32, None, name = 'learning_rate')
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0.5).minimize(self.discriminator_loss)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0.5).minimize(self.generator_loss) 


    def train(self, input_A, input_B, learning_rate):

        generation_A, generation_B, generator_loss, _ = self.sess.run([self.generation_A, self.generation_B, self.generator_loss, self.generator_optimizer], \
            feed_dict = {self.input_A_real: input_A, self.input_B_real: input_B, self.learning_rate: learning_rate})

        discriminator_loss, _ = self.sess.run([self.discriminator_loss, self.discriminator_optimizer], \
            feed_dict = {self.input_A_real: input_A, self.input_B_real: input_B, self.learning_rate: learning_rate, self.input_A_fake: generation_A, self.input_B_fake: generation_B})

        return generator_loss, discriminator_loss


    def test(self, inputs, direction):

        if direction == 'A2B':
            generation = self.sess.run(self.generation_B_test, feed_dict = {self.input_A_test: inputs})
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_A_test, feed_dict = {self.input_B_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        return os.path.join(directory, filename)


if __name__ == '__main__':
    
    model = CycleGAN(input_size = [256, 256, 3])
