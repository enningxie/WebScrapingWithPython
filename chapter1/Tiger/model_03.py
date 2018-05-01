# 05/01
# xz
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import utils


def print_info(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation=True,
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    x = inputs
    if conv_first:
        x = tf.layers.conv2d(
            inputs=x,
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        if batch_normalization:
            x = tf.layers.batch_normalization(inputs=x)
        if activation:
            x = tf.nn.relu(x)
    else:
        if batch_normalization:
            x = tf.layers.batch_normalization(inputs=x)
        if activation:
            x = tf.nn.relu(x)
        x = tf.layers.conv2d(
            inputs=x,
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
    return x


class Resnet(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.n_classes = 94
        self.skip_step = 20  # size for verbose
        self.n_test = 10000  # test set num
        self.training = True
        self.depth = 20

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.process_data(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            img, self.labels = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 224, 224, 4])
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    def inference(self):
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)

        x = resnet_layer(inputs=self.img)  # 1
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=False)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=False,
                                     batch_normalization=False)
                x = tf.add(x, y)
                x = tf.nn.relu(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = tf.layers.average_pooling2d(inputs=x, pool_size=8, strides=8)
        y = tf.layers.Flatten()(x)
        self.logits = tf.layers.dense(inputs=y, units=self.n_classes,
                                      kernel_initializer=tf.keras.initializers.he_normal())

    def loss_op(self):
        '''
        define the loss function.
        :return:
        '''
        with tf.name_scope('loss'):
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            self.loss = tf.reduce_mean(loss_, name='loss')

    def optimize(self):
        '''
        define the optimizer for training op
        :return:
        '''
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)

    def eval(self):
        '''
        count the number of right predictions in a batch
        :return:
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            self.accuracy, self.accuracy_op = tf.metrics.average_precision_at_k(labels=self.labels, predictions=preds, k=94)

    def summary(self):
        '''
        create summaries to write on tensorboard.
        :return:
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        '''
        build the computation graph
        :return:
        '''
        self.get_data()
        self.inference()
        self.loss_op()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}： {1}'.format(step+1, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/vgg16/vgg16_model', step)
        print('Average loss at epoch {0}: {1}'.format(epoch+1, total_loss/n_batches))
        print('Took: {0} seconds.'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        n_batches = 0
        try:
            while True:
                accuracy_batch, _, summaries = sess.run([self.accuracy, self.accuracy_op, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1}'.format(epoch+1, total_correct_preds/n_batches))
        print('Took: {0} seconds.'.format(time.time() - start_time))

    def train(self, n_epochs):
        utils.make_dir('checkpoints')
        utils.make_dir('checkpoints/vgg16')
        writer = tf.summary.FileWriter('./graphs/vgg16', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/vgg16/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.global_step.eval()

            for epoch in range(n_epochs):
                sess.run(tf.local_variables_initializer())
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()


if __name__ == '__main__':
    vgg16 = Resnet()
    vgg16.build()
    vgg16.train(n_epochs=30)
