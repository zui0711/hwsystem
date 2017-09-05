import sys
import os
import math
import time

import numpy as np

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")

import tensorflow as tf

from tensorflow.python.platform import gfile

from models.seq2seq import create_model
from models.LSTM import create_model_lstm
from config.all_params import *
from data_utils import *

from keras.preprocessing import sequence
import keras.backend as K

import cPickle

def seq2seq_train():
    if not gfile.Exists(pjoin(SAVE_DATA_DIR, "nn_models")):
        gfile.MakeDirs(pjoin(SAVE_DATA_DIR, "nn_models"))
        gfile.MakeDirs(pjoin(SAVE_DATA_DIR, "results"))
    print("Preparing dialog data in %s" % FLAGS.data_dir)
    train_data, dev_data, _ = prepare_dialog_data(FLAGS.data_dir, FLAGS.vocab_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        record_file_name = "seq2seq_loss"
        record_file = open(pjoin(FLAGS.results_dir, record_file_name), mode="ab", buffering=0)

        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
        model = create_model(sess, forward_only=False)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
        dev_set = read_data(dev_data)
        train_set = read_data(train_data, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(BUCKETS))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, total_time, loss = 0.0, 0.0, 0.0
        previous_losses = []

        print("start training...")
        # while (True):
        while(model.global_step.eval() <= seq2seq_epoch):
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id, "train")

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if model.global_step.eval() % FLAGS.steps_per_checkpoint == 0:
                print("\nTraining...")
                if loss > 6:
                    print("inf !!!")
                    sys.exit(0)
                total_time = step_time * FLAGS.steps_per_checkpoint
                print ("global step %d learning rate %.4f step-time %.2f total_time %.4f, loss %0.7f" %
                       (model.global_step.eval(), model.learning_rate.eval(), step_time, total_time, loss))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(loss)

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(BUCKETS)):
                    print("\nTesting...")

                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id, "train")
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

                    if eval_loss > 6:
                        print("inf !!!")
                        sys.exit(0)

                    print("eval: bucket %d, loss %0.5f" % (bucket_id, eval_loss))

                # sys.stdout.flush()
                record_file.write("%d\t%.5f\t%.5f\n" % (model.global_step.eval(), loss, eval_loss))

                # Save checkpoint and zero timer and loss.
                if model.global_step.eval() % FLAGS.steps_per_predictpoint == 0:
                    checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    print("saving model...\n")
                step_time, loss = 0.0, 0.0


def seq2seq_predict():
    def _get_test_dataset(encoder_size):
        with open(TEST_DATASET_PATH) as test_fh:
            test_sentences = []
            for line in test_fh.readlines():
                sen = [int(x) for x in line.strip().split()[:encoder_size]]
                if sen:
                    test_sentences.append((sen, []))
        return test_sentences

    bucket_id = 0

    max_len = BUCKETS[bucket_id][0]

    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.txt" % FLAGS.vocab_size)
    vocab, rev_vocab = initialize_vocabulary(vocab_path)

    test_dataset = _get_test_dataset(max_len)
    test_len = len(test_dataset)
    batch_size = FLAGS.predict_batch_size
    batch_num = test_len / batch_size + (0 if test_len % batch_size == 0 else 1)

    file_r = pjoin(SAVE_DATA_DIR, "test/decode.txt")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, forward_only=True)
        # record_file_name = "seq2seq_score"
        # record_file = open(pjoin(FLAGS.results_dir, record_file_name), mode="ab", buffering=0)

        for predict_step in [seq2seq_epoch]:
            predict_path = pjoin(FLAGS.model_dir, "model.ckpt-%d" % predict_step)
            print("Reading model parameters from %s" % predict_path)
            model.saver.restore(sess, predict_path)
            model.batch_size = FLAGS.predict_batch_size

            results_filename = 'results(%d, %d).%d' % (max_len, max_len, model.global_step.eval())
            results_filename_ids = "results(%d, %d).ids%d.%d" % (max_len, max_len, FLAGS.vocab_size, model.global_step.eval())

            results_path = os.path.join(FLAGS.results_dir, results_filename)
            results_path_ids = os.path.join(FLAGS.results_dir, results_filename_ids)

            with open(results_path, 'w') as results_f, open(results_path_ids, 'w') as results_f_ids:
                start_time = time.time()
                for num in xrange(batch_num - 1):
                    this_batch = test_dataset[batch_size*num: batch_size*(num+1)]
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(this_batch, bucket_id, "predict")
                    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

                    for b in xrange(batch_size):
                        outputs = []
                        for en in xrange(max_len):
                            selected_token_id = int(np.argmax(output_logits[en][b]))

                            if selected_token_id == EOS_ID:
                                break
                            else:
                                outputs.append(selected_token_id)

                        output_sentence = ' '.join([rev_vocab[output] for output in outputs])
                        output_sentence_ids = ' '.join([str(output) for output in outputs])

                        # if output_sentence == "":
                        #     print(num, b, output)

                        results_f.write(output_sentence + '\n')
                        results_f_ids.write(output_sentence_ids + "\n")
                    sys.stdout.write("\r")
                    sys.stdout.write("batch %d / %d finish..."%(num + 1, batch_num - 1))
                t = time.time() - start_time
                print("predict: total time = %0.2f, per time = %0.2f" % (t, t / float(batch_num - 1)))

            # print("Predict finish...")

            # print file_r, results_path
            # WER, BLEU = get_score(file_r, results_path)
            # record_file.write("%d\t%.5f\t%.5f\n" % (predict_step, WER, BLEU))

        # record_file.close()



def lstm_train():
    K.clear_session()
    # print(SAVE_DATA_DIR)

    # print('Pad sequences (samples x time)')
    X_train, y_train = load_data_lstm("train")

    X_train = sequence.pad_sequences(X_train, maxlen=LSTM_max_len)
    # X_test = sequence.pad_sequences(X_test, maxlen=LSTM_max_len)
    print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)

    print('\nBuild model...')
    model = create_model_lstm()

    print('\nTrain...')

    print("\n")

    model.fit(X_train, y_train, nb_epoch=lstm_epoch, verbose=1, batch_size=LSTM_batch_size)
    # print "train finish..."
    model.save_weights(pjoin(SAVE_DATA_DIR, "lstm_model_%d.h5"%(lstm_epoch)))
    # cPickle.dump(model, open("lstm_model_%d.h5"%(lstm_epoch)))


def lstm_predict(analysis_mode=False):
    K.clear_session()
    X_test, y_test = load_data_lstm("run")

    X_test = sequence.pad_sequences(X_test, maxlen=LSTM_max_len)
    print('X_test shape:', X_test.shape)

    model = create_model_lstm()
    model.load_weights(pjoin(SAVE_DATA_DIR, "lstm_model_%d.h5"%(lstm_epoch)))

    if not analysis_mode:
        score, acc = model.evaluate(X_test, y_test,
                                    batch_size=LSTM_batch_size)
        print('Test loss: %0.5f' % score)
        print('Test accuracy: %0.5f\n' % acc)

        # record_file.write("%.5f\t%.5f\n" % (score, acc))

    else:
        predict_y = model.predict(X_test)

        ans = {}
        for predict, test in zip(predict_y, y_test):
            if float(predict) < 0.5:
                p_t = (0, int(test))
            else:
                p_t = (1, int(test))
            if p_t in ans:
                ans[p_t] += 1
            else:
                ans[p_t] = 1
        print ans

