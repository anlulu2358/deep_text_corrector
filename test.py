from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import defaultdict

from sklearn.metrics import roc_auc_score, accuracy_score
import nltk

from correct_text import train, decode, decode_sentence, evaluate_accuracy, create_model,\
    get_corrective_tokens, DefaultPTBConfig, DefaultMovieDialogConfig
from text_corrector_data_readers import PTBDataReader, MovieDialogReader

root_data_path = "E:/Deep Learning/deep-text-corrector-master/deep-text-corrector-master/"
train_path = os.path.join(root_data_path, "movie_dialog_train.txt")
val_path = os.path.join(root_data_path, "movie_dialog_val.txt")
test_path = os.path.join(root_data_path, "movie_dialog_test.txt")
model_path = os.path.join(root_data_path, "movie_dialog_model")
config = DefaultMovieDialogConfig()

data_reader = MovieDialogReader(config, train_path)
# train(data_reader, train_path, val_path, model_path)

data_reader = MovieDialogReader(config, train_path, dropout_prob=0.25, replacement_prob=0.25, dataset_copies=1)
corrective_tokens = get_corrective_tokens(data_reader, train_path)
import pickle
with open(os.path.join(root_data_path, "corrective_tokens.pickle"), "wb") as f:
    pickle.dump(corrective_tokens, f)
import pickle
with open(os.path.join(root_data_path, "token_to_id.pickle"), "wb") as f:
    pickle.dump(data_reader.token_to_id, f)
sess = tf.InteractiveSession()
model = create_model(sess, True, model_path, config=config)

decoded = decode_sentence(sess, model, data_reader, "i has a few question", corrective_tokens=corrective_tokens)

decoded = decode_sentence(sess, model, data_reader,
                          "did n't you say that they 're going to develop this revolutionary new thing ...",
                          corrective_tokens=corrective_tokens)

decode_sentence(sess, model, data_reader, "kvothe went to market", corrective_tokens=corrective_tokens, verbose=False)

decode_sentence(sess, model, data_reader, "blablahblah and bladdddd went to market", corrective_tokens=corrective_tokens,
                verbose=False)
decode_sentence(sess, model, data_reader, "do you have book", corrective_tokens=corrective_tokens, verbose=False)

# 4 layers, 40k steps
errors = evaluate_accuracy(sess, model, data_reader, corrective_tokens, test_path)#, max_samples=1000)
# 4 layers, 30k steps
errors = evaluate_accuracy(sess, model, data_reader, corrective_tokens, test_path)#, max_samples=1000)
# 4 layers, 20k steps
errors = evaluate_accuracy(sess, model, data_reader, corrective_tokens, test_path)#, max_samples=1000)
errors = evaluate_accuracy(sess, model, data_reader, corrective_tokens, test_path)#, max_samples=1000)
for decoding, target in errors:
    print("Decoding: " + " ".join(decoding))
    print("Target:   " + " ".join(target) + "\n")                          