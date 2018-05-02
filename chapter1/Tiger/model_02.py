# 04/23
# xz
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import utils


def print_info(t):
    print(t.op.name, ' ', t.get_shape().as_list())


class VGG16(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.n_classes = 94
        self.skip_step = 20  # size for verbose
        self.n_test = 10000  # test set num
        self.training = True

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.process_data(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            img, self.labels = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 224, 224, 4])
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    def inference(self):

        op_list = []

        conv_1 = tf.layers.conv2d(
            inputs=self.img,
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu,
            name='conv_1'
        )

        op_list.append(conv_1)

        pool_1 = tf.layers.max_pooling2d(
            inputs=conv_1,
            pool_size=(2, 2),
            strides=(2, 2),
            name='pool_1'
        )

        op_list.append(pool_1)

        conv_2 = tf.layers.conv2d(
            inputs=pool_1,
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu,
            name='conv_2'
        )

        op_list.append(conv_2)

        pool_2 = tf.layers.max_pooling2d(
            inputs=conv_2,
            pool_size=(2, 2),
            strides=(2, 2),
            name='pool_2'
        )

        op_list.append(pool_2)

        conv_3 = tf.layers.conv2d(
            inputs=pool_2,
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu,
            name='conv_3'
        )

        op_list.append(conv_3)

        conv_4 = tf.layers.conv2d(
            inputs=conv_3,
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
            strides=(1, 1),
            name='conv_4'
        )

        op_list.append(conv_4)

        pool_3 = tf.layers.max_pooling2d(
            inputs=conv_4,
            pool_size=(2, 2),
            strides=(2, 2),
            name='pool_3'
        )

        op_list.append(pool_3)

        conv_5 = tf.layers.conv2d(
            inputs=pool_3,
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu,
            name='conv_5'
        )

        op_list.append(conv_5)

        conv_6 = tf.layers.conv2d(
            inputs=conv_5,
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu,
            name='conv_6'
        )

        op_list.append(conv_6)

        pool_4 = tf.layers.max_pooling2d(
            inputs=conv_6,
            pool_size=(2, 2),
            strides=(2, 2),
            name='pool_4'
        )

        op_list.append(pool_4)

        conv_7 = tf.layers.conv2d(
            inputs=pool_4,
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu,
            name='conv_7'
        )

        op_list.append(conv_7)

        conv_8 = tf.layers.conv2d(
            inputs=conv_7,
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu,
            name='conv_8'
        )

        op_list.append(conv_8)

        pool_5 = tf.layers.max_pooling2d(
            inputs=conv_8,
            pool_size=(2, 2),
            strides=(2, 2),
            name='pool_5'
        )

        op_list.append(pool_5)

        # ---------------------------------------------------

        pool_5_reshape = tf.layers.flatten(pool_5, name='pool_5_reshape')

        op_list.append(pool_5_reshape)

        fc_1 = tf.layers.dense(
            inputs=pool_5_reshape,
            units=1024,
            activation=tf.nn.relu,
            name='fc_1'
        )

        op_list.append(fc_1)

        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=512,
            activation=tf.nn.relu,
            name='fc_2'
        )

        op_list.append(fc_2)

        self.logits = tf.layers.dense(
            inputs=fc_2,
            units=self.n_classes,
            name='logits'
        )

        op_list.append(self.logits)

        for op in op_list:
            print_info(op)
        print("-----------------------let's training.---------------------------")

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
                    print('Loss at step {0}： {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/vgg16/vgg16_model', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/(n_batches*self.batch_size)))
        print('Took: {0} seconds.'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, _, summaries = sess.run([self.accuracy, self.accuracy_op, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1}'.format(epoch, total_correct_preds/self.n_test))
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
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()


if __name__ == '__main__':
    vgg16 = VGG16()
    vgg16.build()
    vgg16.train(n_epochs=30)