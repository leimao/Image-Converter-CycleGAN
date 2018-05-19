
import tensorflow as tf
from module import discriminator, generator_resnet
from utils import l1_loss, l2_loss, cross_entropy_loss

def discriminator(inputs, num_filters, reuse = False, scope_name = 'discriminator'):

def generator_resnet(inputs, num_filters, reuse = False, scope_name = 'generator_resnet'):


class CycleGAN(object):

    def __init__(self, input_size, discriminator = discriminator, generator = generator_resnet, lambda_cycle = 10):

        self.input_size = input_size


        self.discriminator = discriminator
        self.generator = generator
        self.lambda_cycle = lambda_cycle

        self.input_A_real = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A_real')
        self.input_B_real = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B_real')
        # self.input_A_fake and self.input_B_fake will be the placeholders for the generated fake images
        self.input_A_fake = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A_fake')
        self.input_B_fake = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B_fake')


    def build_model(self):

        self.generation_B = self.generator(inputs = self.input_A_real, num_filters = 64, reuse = False, scope_name = 'generator_A2B')
        self.cycle_A = self.generator(inputs = self.generation_B, num_filters = 64, reuse = False, scope_name = 'generator_B2A')

        self.generation_A = self.generator(inputs = self.input_B_real, num_filters = 64, reuse = True, scope_name = 'generator_B2A')
        self.cycle_B = self.generator(inputs = self.generation_A, num_filters = 64, reuse = True, scope_name = 'generator_A2B')

        self.discrimination_A_fake = self.discriminator(inputs = self.generation_A, num_filters = 64, reuse = False, scope_name = 'discriminator_A')
        self.discrimination_B_fake = self.discriminator(inputs = self.generation_B, num_filters = 64, reuse = False, scope_name = 'discriminator_B')

        # Cycle loss
        self.cycle_loss = l1_loss(y = self.input_A_real, y_hat = self.cycle_A) + l1_loss(y = self.input_B_real, y_hat = cycle_B)

        # Generator loss
        # Generator wants to fool discriminator
        self.generator_loss_A2B = l2_loss(y = tf.ones_like(self.discrimination_B_fake), y_hat = self.discrimination_B_fake)
        self.generator_loss_B2A = l2_loss(y = tf.ones_like(self.discrimination_A_fake), y_hat = self.discrimination_A_fake)

        # Merge the two generators and the cycle loss
        self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + self.lambda_cycle * self.cycle_loss

        # Discriminator loss
        self.discrimination_input_A_real = self.discriminator(inputs = self.input_A_real, num_filters = 64, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_input_B_real = self.discriminator(inputs = self.input_B_real, num_filters = 64, reuse = True, scope_name = 'discriminator_B')
        self.discrimination_input_A_fake = self.discriminator(inputs = self.input_A_fake, num_filters = 64, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_input_B_fake = self.discriminator(inputs = self.input_B_fake, num_filters = 64, reuse = True, scope_name = 'discriminator_B')

        # Discriminator wants to classify real and fake correctly
        self.discriminator_loss_input_A_real = l2_loss(y = tf.ones_like(self.discrimination_input_A_real), y_hat = self.discrimination_input_A_real)
        self.discriminator_loss_input_A_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_A_fake), y_hat = self.discrimination_input_A_fake)
        self.discriminator_loss_A = (self.discriminator_loss_input_A_real + self.discriminator_loss_input_A_fake) / 2

        self.discriminator_loss_input_B_real = l2_loss(y = tf.ones_like(self.discrimination_input_B_real), y_hat = self.discrimination_input_B_real)
        self.discriminator_loss_input_A_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_B_fake), y_hat = self.discrimination_input_B_fake)
        self.discriminator_loss_B = (self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake) / 2

        # Merge the two discriminators into one
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B




'''

        self.discriminator_loss_input_B_real = l2_loss(y = tf.ones_like(self.discrimination_input_B_real), y_hat = self.discrimination_input_B_real)
        




        self.discriminator





        # Discriminator loss
        # Minimize (D_A(a) - 1)^2
        self.discriminator_loss_A_1 = l2_loss(y = tf.ones_like(self.discrimination_A_real), y_hat = self.discrimination_A_real)
        # Minimize (D_B(b) - 1)^2
        self.discriminator_loss_B_1 = l2_loss(y = tf.ones_like(self.discrimination_B_real), y_hat = self.discrimination_B_real)
        # Minimize (D_A(G_{B->A}(b)) - 0)^2
        self.discriminator_loss_A_2 = l2_loss(y = tf.zeros_like(self.discrimination_A_fake), y_hat = self.discrimination_A_fake)
        # Minimize (D_B(G_{A->B}(a)) - 0)^2
        self.discriminator_loss_B_2 = l2_loss(y = tf.zeros_like(self.discrimination_B_fake), y_hat = self.discrimination_B_fake)








        # Adversarial loss
        self.discriminator_loss_A = (self.discriminator_loss_A_1 + self.discriminator_loss_A_2) / 2
        self.discriminator_loss_B = (self.discriminator_loss_B_1 + self.discriminator_loss_B_2) / 2

        # Generator loss
        # Minimize (D_A(G_{B->A}(b)) - 1)^2
        self.generator_loss_A = l2_loss(y = tf.ones_like(self.discrimination_A_fake), y_hat = self.discrimination_A_fake)
        # Minimize (D_B(G_{A->B}(a)) - 1)^2
        self.generator_loss_B = l2_loss(y = tf.ones_like(self.discrimination_B_fake), y_hat = self.discrimination_B_fake)




        # 


















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
'''