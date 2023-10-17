import time
from models.ops import *
from models.utils import *
import shutil

class ResNet(object):
    def __init__(self, sess, dataset,checkpoint_dir,log_dir,res_n,batch_size,lr,tx,tx_l,tes,tes_y):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = dataset

        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist' :
            # self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            # print(self.train_x,self.train_y.shape)
            self.train_x = tx
            self.train_y = tx_l
            self.test_x = tes
            self.test_y = tes_y
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200


        # self.checkpoint_dir = arg.checkpoint_dir
        self.checkpoint_dir = checkpoint_dir
        # self.log_dir = args.log_dir
        self.log_dir = log_dir

        # self.res_n = args.res_n
        self.res_n = res_n

        self.max=0

        # self.epoch = args.epoch
        self.epoch = 100

        self.batch_size = batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = 0.1
        self.dropout_keep_prob = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), tuple(),
                                                             name='dropout_keep_prob')
        # self.batch_size = args.batch_size
        # self.iteration = len(self.train_x) // self.batch_size
        #
        # self.init_lr = args.lr


    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("network", reuse=reuse) as scope:

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x1 = global_avg_pooling(x)



            y = fully_conneted(x1, units=self.label_dim, scope='logit')

            x=  tf.nn.dropout(y, 1.0)##change

            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.X=tf.placeholder(tf.float32, shape=(None, self.img_size, self.img_size, self.c_dim), name='X')

        self.train_inptus = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [None, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [None, self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        # with tf.GradientTape() as tape:
        #     self.train_logits = self.network(self.train_inptus)
        # self.grad=tape.gradient(self.train_logits,self.train_inptus)
        # print("build_grad {}".format(self.grad))

        self.train_logits = self.network(self.train_inptus)


        self.test_logits = self.network(self.test_inptus, is_training=False, reuse=True)
        self.my_out1=self.network(self.X, is_training=False, reuse=True)

        self.train_loss, self.train_accuracy = classification_loss(logit=self.train_logits, label=self.train_labels)
        self.test_loss, self.test_accuracy = classification_loss(logit=self.test_logits, label=self.test_labels)
        self.grad = tf.gradients(self.train_loss, self.train_logits)
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss


        """ Training """
        self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    def build_model_2(self):
        """ Graph Input """
        # self.X = tf.placeholder(tf.float32, shape=(None, self.img_size, self.img_size, self.c_dim), name='X')
        #
        # self.train_inptus = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.c_dim],
        #                                    name='train_inputs')
        # self.train_labels = tf.placeholder(tf.float32, [None, self.label_dim], name='train_labels')
        #
        # self.test_inptus = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.c_dim],
        #                                   name='test_inputs')
        # self.test_labels = tf.placeholder(tf.float32, [None, self.label_dim], name='test_labels')
        #
        # self.lr = tf.placeholder(tf.float32, name='learning_rate')
        #
        # """ Model """
        # self.train_logits = self.network(self.train_inptus)
        # self.test_logits = self.network(self.test_inptus, is_training=False, reuse=True)
        self.my_out = self.network(self.X, is_training=False, reuse=tf.AUTO_REUSE)

    def normalize_scores(self):
        with tf.name_scope('output'):
            self.prob                = tf.nn.softmax(self.my_out, name="prob")
            # self.predictions         = tf.argmax(self.prob, axis = 1)
            # self.predictions_one_hot = tf.one_hot(self.predictions, self.num_classes)


    ##################################################################################
    # Train
    ##################################################################################

    def train_test(self,test_x,test_y,counter):
        out = []
        for idx in range(len(test_x) // self.batch_size):
            test_feed_dict = {
                # self.X:test_x[idx * self.batch_size:(idx + 1) * self.batch_size],
                self.test_inptus: test_x[idx * self.batch_size:(idx + 1) * self.batch_size],
                self.test_labels: test_y[idx * self.batch_size:(idx + 1) * self.batch_size]
            }

            summary_str, test_loss, test_accuracy = self.sess.run(
                [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
            self.writer.add_summary(summary_str, counter)
            # test_logits = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
            out.append(test_accuracy)

        out = np.mean(out)
        return out

    def train(self):
        print ("xunlianjichicun:",self.train_x.shape)
        count_evaluate=0
        collect_grad=[]
        self.max=0
        no_improvement = 0
        exit=False

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # shutil.rmtree(self.checkpoint_dir, ignore_errors=True, onerror=None)
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)



        # restore check-point if it exits
        # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        could_load=False##change
        if could_load:
            epoch_lr = self.init_lr
            # start_epoch = (int)(checkpoint_counter / self.iteration)
            # start_batch_id = checkpoint_counter - start_epoch * self.iteration
            # counter = checkpoint_counter

            # if start_epoch >= int(self.epoch * 0.75) :
            #     epoch_lr = epoch_lr * 0.01
            # elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
            #     epoch_lr = epoch_lr * 0.1
            # print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                count_evaluate+=1
                batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus : self.test_x,
                    self.test_labels : self.test_y
                }


                # update network

                #     self.train_logits = self.network(self.train_inptus)
                # self.grad=tape.gradient(self.train_logits,self.train_inptus)
                # print("build_grad {}".format(self.grad))
                _, summary_str, train_loss, train_accuracy,grad1= self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy,self.grad], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                # summary_str, test_loss, test_accuracy = self.sess.run(
                #     [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                # self.writer.add_summary(summary_str, counter)
                if count_evaluate % self.iteration==0:
                    count_evaluate=0
                    tt = self.train_test(self.test_x, self.test_y, counter)
                    counter += 1
                    if self.max<tt or self.max==0:
                        self.max=tt
                        self.save(self.checkpoint_dir, counter)
                        print("save model")
                        no_improvement=0
                    else:
                        no_improvement+=1
                    # display training status
                        if (no_improvement % 20) == 0:
                            if train_accuracy<0.7:
                                no_improvement=0
                            else:
                                exit=True

                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                          % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, tt, epoch_lr))



            # print("tudu {}".format(grad1))
            collect_grad.append(grad1[0])

            if exit:
                print ("Number of epochs processed: {} in acc {}".format(epoch + 1,tt))
                break

                    # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
        gard_all=np.concatenate(collect_grad)
        np.save(os.path.join(self.log_dir, 'gard.npy'), gard_all)
        # save model for final step
        # self.save(self.checkpoint_dir, counter)
        return self.max

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: self.test_x,
            self.test_labels: self.test_y
        }


        test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))




    def get_graph(self):
        return tf.get_default_graph()

    def change_train(self,tx,ty,sess,logdir):
        self.train_x = tx
        self.train_y = ty
        self.sess=sess
        self.iteration=len(self.train_x) // self.batch_size
        self.checkpoint_dir=logdir
        self.log_dir=logdir

    def do_my_test(self,test_x,test_y):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        out = []

        for idx in range(len(test_x) // self.batch_size):
            test_feed_dict = {
                self.test_inptus: test_x[idx * self.batch_size:(idx + 1) * self.batch_size],
                self.test_labels: test_y[idx * self.batch_size:(idx + 1) * self.batch_size]
            }

            test_logits = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
            # summary_str, test_loss, test_accuracy = self.sess.run(
            #     [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
            # self.writer.add_summary(summary_str, counter)
            out.append(test_logits)
        out = np.mean(out)

        # test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(out))
        return float(out)

    def get_my_label(self,x):
        left=False
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS in get my label")
        else:
            print(" [!] Load failed...in get my label")


        out=[]
        if len(x) % self.batch_size!=0:
            left=True
        for idx in range(len(x)//self.batch_size):
            test_feed_dict = {
                self.X: x[idx*self.batch_size:(idx+1)*self.batch_size],

            }

            test_logits = self.sess.run(self.my_out1, feed_dict=test_feed_dict)

            out.append(test_logits)
        if left:
            test_feed_dict1 = {
                self.X: x[(len(x)//self.batch_size) * self.batch_size:],
            }
            test_logit1 = self.sess.run(self.my_out1, feed_dict=test_feed_dict1)
            out.append(test_logit1)

        out=np.concatenate(out)

        return out
        # print("test_accuracy: {}".format(test_accuracy))

    def get_my_label_without_reload(self,x):
        left = False
        out=[]
        if len(x) % self.batch_size!=0:
            left=True
        for idx in range(len(x)//self.batch_size):
            test_feed_dict = {
                self.X: x[idx*self.batch_size:(idx+1)*self.batch_size],

            }

            test_logits = self.sess.run(self.my_out1, feed_dict=test_feed_dict)

            out.append(test_logits)

        if left:
            test_feed_dict1 = {
                self.X: x[(len(x)//self.batch_size) * self.batch_size:],
            }
            test_logit1 = self.sess.run(self.my_out1, feed_dict=test_feed_dict1)
            out.append(test_logit1)
        out=np.concatenate(out)

        return out

