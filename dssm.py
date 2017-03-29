import pickle
import random
import time
import sys
import numpy as np
import linecache
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', '/tmp/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 900000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 10, "Number of steps in one epoch.")
flags.DEFINE_integer('pack_size', 2000, "Number of batches in one pickle pack.")
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

start = time.time()

TRIGRAM_D = 50000

NEG=10
BS = 1000

L1_N = 400
L2_N = 120

query_in_shape = np.array([BS, TRIGRAM_D], np.int64)
doc_in_shape = np.array([BS, TRIGRAM_D], np.int64)
label_in_shape = np.array([BS, 1], np.int64)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


with tf.name_scope('input'):
    # Shape [BS, TRIGRAM_D].
    query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')
    # Shape [BS, TRIGRAM_D]
    doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')
    label_batch = tf.sparse_placeholder(tf.float32, shape=label_in_shape, name='LabelBatch')

with tf.name_scope('L1'):
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')

    # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
    doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

    query_l1_out = tf.nn.relu(query_l1)
    doc_l1_out = tf.nn.relu(doc_l1)

with tf.name_scope('L2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
    query_y = tf.nn.relu(query_l2)
    doc_y = tf.nn.relu(doc_l2)

with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce 50 FD-
    temp = tf.tile(doc_y, [1, 1])

    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat(0,
                          [doc_y,
                           tf.slice(temp, [rand, 0], [BS - rand, -1]),
                           tf.slice(temp, [0, 0], [rand, -1])])

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.mul(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.mul(query_norm, doc_norm)

    cos_sim_raw = tf.truediv(prod, norm_prod)
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    prob = tf.nn.softmax((cos_sim))
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
    tf.scalar_summary('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

# with tf.name_scope('Accuracy'):
#     correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.scalar_summary('accuracy', accuracy)

merged = tf.merge_all_summaries()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.scalar_summary('average_loss', average_loss)


def data_iterator(file):
    """ A simple data iterator """
    index_start = 1
    while True:
        u_indices = []      
        n_indices = []
        labels = []
        for i in range(0, BS):
            line = linecache.getline(file, index_start+i)
            if (not line):
                index_start = -i
                continue
            label, user, news = line.strip().split('\3')
            labels.append(label)
            us = user.split('\t')
            ns = news.split('\t')
            for j in range(1, len(us)):
                u_indices.append([i,us[j]])
            for j in range(1, len(ns)):
                n_indices.append([i, ns[j]])
        u_values = np.ones(len(u_indices))
        n_values = np.ones(len(n_indices))
        u_indices = np.array(u_indices)
        n_indices = np.array(n_indices)
        label = np.array(labels)
        #print u_indices
        #print n_indices
        #print u_values
        #print n_values
        #print label
        query_in = tf.SparseTensorValue(u_indices, u_values, [BS, TRIGRAM_D])
        doc_in =  tf.SparseTensorValue(n_indices, n_values, [BS, TRIGRAM_D])

        index_start += BS
        yield query_in, doc_in, index_start #, label

config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
#if not FLAGS.gpu:
#config = tf.ConfigProto(device_count= {'GPU' : 0})

iter_ = data_iterator("train")
with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test', sess.graph)

    # Actual execution
    start = time.time()
    # fp_time = 0
    # fbp_time = 0
    for step in range(FLAGS.max_steps):
            # # setup toolbar
            # sys.stdout.write("[%s]" % (" " * toolbar_width))
            # #sys.stdout.flush()
            # sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['



        # t1 = time.time()
        # sess.run(loss, feed_dict = feed_dict(True, batch_idx))
        # t2 = time.time()
        # fp_time += t2 - t1
        # #print(t2-t1)
        # t1 = time.time()
        query_in, doc_in, index_start = iter_.next()
        sess.run(train_step, feed_dict={query_batch: query_in, doc_batch: doc_in}) #, label_batch:label})
        #print(sess.run(prob, feed_dict={query_batch: query_in, doc_batch: doc_in, label_batch:label}))
        #print
        #print(sess.run(weight1))
        #print
        #print(sess.run(bias1))
        #break
        # t2 = time.time()
        # fbp_time += t2 - t1
        # #print(t2 - t1)
        # if batch_idx % 2000 == 1999:
        #     print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
        #        (fp_time / step, fbp_time / step))

        print index_start
        if step % FLAGS.epoch_steps == 0:
            end = time.time()
            epoch_loss = 0
            
            loss_v = sess.run(loss, feed_dict={query_batch: query_in, doc_batch: doc_in})
            epoch_loss += loss_v

            
            train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(train_loss, step + 1)

            # print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
            #        (fp_time / step, fbp_time / step))
            #
            print ("\nMiniBatch: %-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs | File ptr: %d" %
                    (step, epoch_loss, end - start, index_start))
