import tensorflow as tf
import numpy as np
import os
import datetime
import time

from text_cnn import TextCNN
import data_helpers
import utils
from logger import Logger
from configure import FLAGS

from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def train():
    with tf.device('/cpu:0'):
        x_text, y, label, pos1, pos2 = data_helpers.load_data_and_labels(FLAGS.train_path)
    with tf.device('/cpu:0'):
        test_text, test_y, test_label, test_pos1, test_pos2 = data_helpers.load_data_and_labels(FLAGS.test_path)
    
    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = MAX_SENTENCE_LENGTH
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    vocab_processor.fit(x_text + test_text)
    train_x = np.array(list(vocab_processor.transform(x_text)))
    test_x = np.array(list(vocab_processor.transform(test_text)))
    train_text = np.array(x_text)
    test_text = np.array(test_text)
    print("\nText Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("train_x = {0}".format(train_x.shape))
    print("train_y = {0}".format(y.shape))
    print("test_x = {0}".format(test_x.shape))
    print("test_y = {0}".format(test_y.shape))

    # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
    # [95 96 97 98 99 100 101 999 999 999 ... 999]
    # =>
    # [11 12 13 14 15  16  21  17  17  17 ...  17]
    # dimension = MAX_SENTENCE_LENGTH
    pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    pos_vocab_processor.fit(pos1 + pos2 + test_pos1 + test_pos2)
    train_p1 = np.array(list(pos_vocab_processor.transform(pos1)))
    train_p2 = np.array(list(pos_vocab_processor.transform(pos2)))
    test_p1 = np.array(list(pos_vocab_processor.transform(test_pos1)))
    test_p2 = np.array(list(pos_vocab_processor.transform(test_pos2)))
    print("\nPosition Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    print("train_p1 = {0}".format(train_p1.shape))
    print("test_p1 = {0}".format(test_p1.shape))
    print("")

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
            sequence_length=train_x.shape[1],
            num_classes=y.shape[1],
            text_vocab_size=len(vocab_processor.vocabulary_),
            text_embedding_size=FLAGS.text_embedding_dim,
            pos_vocab_size=len(pos_vocab_processor.vocabulary_),
            pos_embedding_size=FLAGS.pos_embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(cnn.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("\nWriting to {}\n".format(out_dir))

            # Logger
            logger = Logger(out_dir)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            pos_vocab_processor.save(os.path.join(out_dir, "pos_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.embedding_path:
                pretrain_W = utils.load_glove(FLAGS.embedding_path, FLAGS.text_embedding_dim, vocab_processor)
                sess.run(cnn.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")
            
            # adding attention weight
            # w1, w2 = data_helpers.word_entity_sim(train_x, train_e1, train_e2, vocab_processor, pretrain_W)
            # w1 = data_helpers.softmax(w1)
            # w2 = data_helpers.softmax(w2)
            # input_features = (w1*train_x + w2*train_x)/2
            # Generate batches
            batches = data_helpers.batch_iter(list(zip(train_x, pos1, pos2, label, y)),
                                              FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            for batch in batches:
                x_batch, p1_batch, p2_batch, y_label, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    cnn.input_text: x_batch,
                    cnn.input_p1: p1_batch,
                    cnn.input_p2: p2_batch,
                    cnn.input_label: y_label,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # Generate batches
                    test_batches = data_helpers.batch_iter(list(zip(test_text, test_pos1, test_pos2, test_label, test_y)),
                                                           FLAGS.batch_size, 1, shuffle=False)
                    # Training loop. For each batch...
                    losses = 0.0
                    accuracy = 0.0
                    predictions = []
                    iter_cnt = 0
                    for test_batch in test_batches:
                        test_bx, test_bp1, test_bp2, test_lbl, test_by = zip(*test_batch)
                        feed_dict = {
                            cnn.input_text: test_bx,
                            cnn.input_p1: test_bp1,
                            cnn.input_p2: test_bp2,
                            cnn.input_label: test_lbl,
                            cnn.input_y: test_by,
                            cnn.dropout_keep_prob: 1.0
                        }
                        loss, acc, pred = sess.run(
                            [cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
                        losses += loss
                        accuracy += acc
                        predictions += pred.tolist()
                        iter_cnt += 1
                    losses /= iter_cnt
                    accuracy /= iter_cnt
                    predictions = np.array(predictions, dtype='int')

                    logger.logging_eval(step, loss, accuracy, predictions)

                    # Model checkpoint
                    if best_f1 < logger.best_f1:
                        best_f1 = logger.best_f1
                        path = saver.save(sess, checkpoint_prefix+"-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
