
import tensorflow as tf
from module import discriminator, generator_resnet
from utils import l1_loss, l2_loss, cross_entropy_loss

def discriminator(inputs, num_filters, reuse = False, scope_name = 'discriminator'):

def generator_resnet(inputs, num_filters, reuse = False, scope_name = 'generator_resnet'):


class CycleGAN(object):

    def __init__(self, input_size, discriminator = discriminator, generator = generator_resnet):

        self.input_size = input_size


        self.discriminator = discriminator
        self.generator = generator
        self.


        self.input_A = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A')
        self.input_B = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B')

    def build_model(self):

        self.generation_B = self.generator(inputs = self.input_A, num_filters = 64, reuse = False, scope_name = 'generator_A2B')
        self.cycle_A = self.generator(inputs = self.generation_B, num_filters = 64, reuse = False, scope_name = 'generator_B2A')

        self.generation_A = self.generator(inputs = self.input_B, num_filters = 64, reuse = True, scope_name = 'generator_B2A')
        self.cycle_B = self.generator(inputs = self.generation_A, num_filters = 64, reuse = True, scope_name = 'generator_A2B')

        self.discrimination_A_fake = self.discriminator(inputs = self.generation_A, num_filters = 64, reuse = False, scope_name = 'discriminator_A')
        self.discrimination_B_fake = self.discriminator(inputs = self.generation_B, num_filters = 64, reuse = False, scope_name = 'discriminator_B')
        self.discrimination_A_real = self.discriminator(inputs = self.input_A, num_filters = 64, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_B_real = self.discriminator(inputs = self.input_B, num_filters = 64, reuse = True, scope_name = 'discriminator_B')

        # Discriminator loss
        # Minimize (D_A(a) - 1)^2
        self.discriminator_loss_A_1 = l2_loss(y = tf.ones_like(self.discrimination_A), y_hat = self.discrimination_A)










        self.discriminator_loss_A_1 = l2_loss(y = tf.ones_like(self.discrimination_A), y_hat = self.discrimination_A)
        # Minimize (D_B(b) - 1)^2
        self.discriminator_loss_B_1 = l2_loss(y = tf.ones_like(self.discrimination_B), y_hat = self.discrimination_B)
        # Minimize (D_A(G_{B->A}(b)))
        self.discriminator_loss_A_2 = l2_loss(y = tf.ones_like(self.cycle_A), y_hat = self.cycle_A)
        self.discriminator_loss_B_2 = l2_loss(y = tf.ones_like(self.cycle_B), y_hat = self.cycle_B)

        self.discriminator_loss_A = (self.discriminator_loss_A_1 + self.discriminator_loss_A_2) / 2
        self.discriminator_loss_B = (self.discriminator_loss_B_1 + self.discriminator_loss_B_2) / 2

        # Generator loss

        self.generator_loss_A2B = 










class CNN(object):

    def __init__(self, input_size, num_classes, optimizer):

        self.num_classes = num_classes
        self.input_size = input_size
        self.optimizer = optimizer

        self.learning_rate = tf.placeholder(tf.float32, shape = [], name = 'learning_rate')
        self.dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')
        self.input = tf.placeholder(tf.float32, [None] + self.input_size, name = 'input')
        self.label = tf.placeholder(tf.float32, [None, self.num_classes], name = 'label')
        self.output = self.network_initializer()
        self.loss = self.loss_initializer()
        self.optimizer = self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())