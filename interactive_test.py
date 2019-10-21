# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Inference
'''

import os

import tensorflow as tf

from data_load import get_batch, input_fn, generator_fn
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, load_hparams
import logging
import sentencepiece as spm


def get_batch_single(sent1, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    sent1: source sentence string.
    sent2: target sentence string
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''

    sents = [sent1.strip().encode('utf-8')]
    #batches = input_fn(sents, sents, vocab_fpath, batch_size, shuffle=shuffle)
    batches = generator_fn(sents, sents, vocab_fpath)
    return list(batches)[0]


logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)

logging.info("# Load model")

logging.info("# Load trained bpe model")
sp = spm.SentencePieceProcessor()
sp.Load("iwslt2016/segmented/bpe.model")

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_  # None: ckpt is a file. otherwise dir.
    # tmp_text = "我 是 何 杰"
    # tmp_pieces = sp.EncodeAsPieces(tmp_text)
    # tmp_batches, num_tmp_batches, num_tmp_samples = get_batch_single(" ".join(tmp_pieces) + "\n",
    #                                                                     hp.vocab, 1, shuffle=False)
    # tmp_iter = tf.data.Iterator.from_structure(tmp_batches.output_types, tmp_batches.output_shapes)
    # tmp_x, tmp_y = tmp_iter.get_next()
    # print(f"xs is {tmp_x}")
    # print(f"ys is {tmp_y}")
    xs = tf.placeholder(dtype=tf.int32, shape=(None, None)), tf.placeholder(dtype=tf.int32, shape=(None,))
    ys = tf.placeholder(dtype=tf.int32, shape=(None, None)), tf.placeholder(dtype=tf.int32, shape=(None, None)), \
         tf.placeholder(dtype=tf.int32, shape=(None,))

    m = Transformer(hp)
    y_hat, _ = m.eval(xs, ys)
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    while True:
        raw_text = input("Model prompt: ")
        if raw_text == 'EOF':
            break
        pieces = sp.EncodeAsPieces(raw_text)
        (x0, x1), (y0, y1, y2) = get_batch_single(" ".join(pieces) + "\n", hp.vocab, 1, shuffle=False)
        import numpy as np
        x = [[x0], [x1]]
        x = [np.array(x) for x in x]
        y = [[y0], [y1], [y2]]
        y = [np.array(y) for y in y]
        print(f"x is {x}")
        print(f"y is {y}")
        out = sess.run(y_hat, feed_dict={xs: x, ys: y})

        sent = "".join(m.idx2token[idx] for idx in out.tolist()[0])
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ")  # remove bpe symbols
        out_sentence = sent.strip()
        print(out_sentence)

