import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle
import sys
import time
import utils
import config
import logging
import nn_layers
import lasagne.layers as L
from nn_layers import QuerySliceLayer
from nn_layers import AttentionSumLayer
from nn_layers import GatedAttentionLayerWithQueryAttention
from IPython import embed

def gen_examples(x1, x2, x3, y, batch_size, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * 4 + k] for t in minibatch for k in range(4)]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2)
        mb_x3, mb_mask3 = utils.prepare_data(mb_x3)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y))
    return all_ex


def build_fn(args, embeddings):
    """
        Build training and testing functions.
    """
    in_x1 = T.imatrix('x1')
    in_x2 = T.imatrix('x2')
    in_x3 = T.imatrix('x3')
    in_mask1 = T.matrix('mask1')
    in_mask2 = T.matrix('mask2')
    in_mask3 = T.matrix('mask3')
    in_y = T.ivector('y')

    l_in1 = lasagne.layers.InputLayer((None, None), in_x1)
    l_mask1 = lasagne.layers.InputLayer((None, None), in_mask1)
    l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, args.vocab_size,
                                           args.embedding_size, W=embeddings)

    l_in2 = lasagne.layers.InputLayer((None, None), in_x2)
    l_mask2 = lasagne.layers.InputLayer((None, None), in_mask2)
    l_emb2 = lasagne.layers.EmbeddingLayer(l_in2, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    l_in3 = lasagne.layers.InputLayer((None, None), in_x3)
    l_mask3 = lasagne.layers.InputLayer((None, None), in_mask3)
    l_emb3 = lasagne.layers.EmbeddingLayer(l_in3, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    if not args.tune_embedding:
        l_emb1.params[l_emb1.W].remove('trainable')
        l_emb2.params[l_emb2.W].remove('trainable')
        l_emb3.params[l_emb3.W].remove('trainable')

    args.rnn_output_size = args.hidden_size * 2 if args.bidir else args.hidden_size
    if args.model == "GA":
        l_d = l_emb1
        # NOTE: This implementation slightly differs from the original GA reader. Specifically:
        # 1. The query GRU is shared across hops.
        # 2. Dropout is applied to all hops (including the initial hop).
        # 3. Gated-attention is applied at the final layer as well.
        # 4. No character-level embeddings are used.

        l_q = nn_layers.stack_rnn(l_emb2, l_mask2, 1, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='q',
                                       rnn_layer=args.rnn_layer)
        q_length = nn_layers.LengthLayer(l_mask2)
        network2 = QuerySliceLayer([l_q, q_length])
        for layer_num in xrange(args.num_GA_layers):
            l_d = nn_layers.stack_rnn(l_d, l_mask1, 1, args.hidden_size,
                                      grad_clipping=args.grad_clipping,
                                      dropout_rate=args.dropout_rate,
                                      only_return_final=False,
                                      bidir=args.bidir,
                                      name='d' + str(layer_num),
                                      rnn_layer=args.rnn_layer)
            l_d = GatedAttentionLayerWithQueryAttention([l_d, l_q, l_mask2])
        network1 = l_d
    else:
        assert args.model is None
        network1 = nn_layers.stack_rnn(l_emb1, l_mask1, args.num_layers, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=(args.att_func == 'last'),
                                       bidir=args.bidir,
                                       name='d',
                                       rnn_layer=args.rnn_layer)

        network2 = nn_layers.stack_rnn(l_emb2, l_mask2, args.num_layers, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=True,
                                       bidir=args.bidir,
                                       name='q',
                                       rnn_layer=args.rnn_layer)
    if args.att_func == 'mlp':
        attx = nn_layers.MLPAttentionLayer([network1, network2], args.rnn_output_size,
                                          mask_input=l_mask1)
    elif args.att_func == 'bilinear':
        attx = nn_layers.BilinearAttentionLayer([network1, network2], args.rnn_output_size,
                                               mask_input=l_mask1)
    elif args.att_func == 'avg':
        attx = nn_layers.AveragePoolingLayer(network1, mask_input=l_mask1)
    elif args.att_func == 'last':
        attx = network1
    elif args.att_func == 'dot':
        attx = nn_layers.DotProductAttentionLayer([network1, network2], mask_input=l_mask1)
    else:
        raise NotImplementedError('att_func = %s' % args.att_func)
    network3_un = nn_layers.stack_rnn(l_emb3, l_mask3, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='o',
                                   rnn_layer=args.rnn_layer)
    network3 = lasagne.layers.ReshapeLayer(network3_un, (in_x1.shape[0], 4, args.rnn_output_size))
    option1 = lasagne.layers.SliceLayer(network3, indices=0,axis=1)
    option2 = lasagne.layers.SliceLayer(network3, indices=1,axis=1)
    option3 = lasagne.layers.SliceLayer(network3, indices=2,axis=1)
    option4 = lasagne.layers.SliceLayer(network3, indices=3,axis=1)
    #orthogon_rep = nn_layers.orthogonalize([att, network2, option1, option2, option3, option4],args.rnn_output_size)
    att = attx
    for ii in xrange(args.no_elem):
    	att = nn_layers.orthogonalize([att, network2, option1, option2, option3, option4], args.rnn_output_size)
    network = nn_layers.BilinearDotLayer([network3, att], args.rnn_output_size)
    if args.pre_trained is not None:
        dic = utils.load_params(args.pre_trained)
        if args.end_to_end:
            lasagne.layers.set_all_param_values(network, dic['params'])
        else:
            if args.test_only:
                lasagne.layers.set_all_param_values(network, dic['params'])
            else:
                lasagne.layers.set_all_param_values([network3, attx], dic['params'])
        del dic['params']
        logging.info('Loaded pre-trained model: %s' % args.pre_trained)
        for dic_param in dic.iteritems():
            logging.info(dic_param)
    
    logging.info('#params: %d' % lasagne.layers.count_params(network, trainable=True))
    logging.info('#fixed params: %d' % lasagne.layers.count_params(network, trainable=False))
    for layer in lasagne.layers.get_all_layers(network):
        logging.info(layer)

    # Test functions
    test_prob = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))
    test_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_y], [acc, test_prediction], on_unused_input='warn')
    
    sliced = lasagne.layers.SliceLayer(att, indices=0, axis=0)
    option_1 = lasagne.layers.SliceLayer(network3, indices=0,axis=1)
    option_2 = lasagne.layers.SliceLayer(network3, indices=1,axis=1)
    option_3 = lasagne.layers.SliceLayer(network3, indices=2,axis=1)
    option_4 = lasagne.layers.SliceLayer(network3, indices=3,axis=1)
    # Train functions
    train_prediction = lasagne.layers.get_output(network)
    train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, in_y).mean()
    # TODO: lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    #params = lasagne.layers.get_all_params(network)#, trainable=True)

    all_params = lasagne.layers.get_all_params(network)
    if args.end_to_end:
        train_params = all_params
    else:
        freeze_params = lasagne.layers.get_all_params([network3, attx])
        train_params = list(set(all_params) - set(freeze_params))
    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, train_params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(loss, train_params, learning_rate=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, train_params, learning_rate=args.learning_rate)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
    
    train_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_y],
                               loss,updates=updates, on_unused_input='warn')
    my_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_y],[train_prediction, lasagne.layers.get_output(option_1,deterministic=True),lasagne.layers.get_output(option_2,deterministic=True),lasagne.layers.get_output(option_3,deterministic=True),lasagne.layers.get_output(option_4,deterministic=True),lasagne.layers.get_output(sliced,deterministic=True),lasagne.layers.get_output(att, deterministic=True), lasagne.layers.get_output(l_q,deterministic=True),lasagne.layers.get_output(network, deterministic=True),lasagne.layers.get_output(network3,deterministic=True),lasagne.layers.get_output(network2,deterministic=True),lasagne.layers.get_output(network1,deterministic=True),lasagne.layers.get_output(l_emb1,deterministic=True),lasagne.layers.get_output(l_emb2,deterministic=True),lasagne.layers.get_output(l_emb3,detrministic=True),loss], on_unused_input='warn')
    return train_fn, test_fn, train_params, all_params, my_fn


def eval_acc(test_fn, all_examples):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = 0
    n_examples = 0
    prediction = []
    for x1, mask1, x2, mask2, x3, mask3, y in all_examples:
        tot_acc, pred = test_fn(x1, mask1, x2, mask2, x3, mask3, y)
        acc += tot_acc
        prediction += pred.tolist()
        n_examples += len(x1)
    return acc * 100.0 / n_examples, prediction


def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')
    question_belong = []
    if args.debug:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, 100, relabeling=args.relabeling)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, 100, relabeling=args.relabeling, question_belong=question_belong)
    else:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, relabeling=args.relabeling)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling, question_belong=question_belong)
    #embed()
    args.num_train = len(train_examples[0])
    args.num_dev = len(dev_examples[0])

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    #word_dict = utils.build_dict(train_examples[0] + train_examples[1] + train_examples[2], args.max_vocab_size)
    word_dict = pickle.load(open("../obj/dict.pkl", "rb"))
    #embed()
    logging.info('-' * 50)
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.info('Compile functions..')
    train_fn, test_fn, train_params, all_params, my_fn = build_fn(args, embeddings)
    logging.info('Done.')
    logging.info('-' * 50)
    logging.info(args)

    logging.info('-' * 50)
    logging.info('Intial test..')
    dev_x1, dev_x2, dev_x3, dev_y = utils.vectorize(dev_examples, word_dict, sort_by_len=not args.test_only, concat=args.concat)
    word_dict_r = {}
    word_dict_r[0] = "unk"
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_y, args.batch_size, args.concat)
    dev_acc, pred = eval_acc(test_fn, all_dev)
    logging.info('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc
    if args.test_only:
        return
    utils.save_params(args.model_file, all_params, epoch=0, n_updates=0)

    # Training
    logging.info('-' * 50)
    logging.info('Start training..')
    train_x1, train_x2, train_x3, train_y = utils.vectorize(train_examples, word_dict, concat=args.concat)
    assert len(train_x1) == args.num_train
    start_time = time.time()
    n_updates = 0
    #embed()
    all_train = gen_examples(train_x1, train_x2, train_x3, train_y, args.batch_size, args.concat)
    for epoch in range(args.num_epoches):
        np.random.shuffle(all_train)
        for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y) in enumerate(all_train):
            #embed()
            train_loss = train_fn(mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y)
            #pred,s_att, att, l_q, net, net3, net2, net1, emb1, emb2, emb3, train_loss = my_fn(mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y)
            #embed()
            if idx % 100 == 0:
                logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1

            if n_updates % args.eval_iter == 0:
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                                            [train_x2[k] for k in samples],
                                            [train_x3[k * 4 + o] for k in samples for o in range(4)],
                                            [train_y[k] for k in samples],
                                            args.batch_size, args.concat)
                acc, pred = eval_acc(test_fn, sample_train)
                logging.info('Train accuracy: %.2f %%' % acc)
                dev_acc, pred = eval_acc(test_fn, all_dev)
                logging.info('Dev accuracy: %.2f %%' % dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, dev_acc))
                    utils.save_params(args.model_file, all_params, epoch=epoch, n_updates=n_updates)


if __name__ == '__main__':
    args = config.get_args()
    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))

    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    if args.rnn_type == 'lstm':
        args.rnn_layer = lasagne.layers.LSTMLayer
    elif args.rnn_type == 'gru':
        args.rnn_layer = lasagne.layers.GRULayer
    else:
        raise NotImplementedError('rnn_type = %s' % args.rnn_type)

    if args.embedding_file is not None:
        dim = utils.get_dim(args.embedding_file)
        if (args.embedding_size is not None) and (args.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                             (args.embedding_size, args.embedding_file, dim))
        args.embedding_size = dim
    elif args.embedding_size is None:
        raise RuntimeError('Either embedding_file or embedding_size needs to be specified.')

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))
    main(args)
