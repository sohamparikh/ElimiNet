model=../obj/model_GA_64_2hops.pkl.gz
gpu=gpu1
option_suffix="-num_GA_layers 2 -hidden_size 64 -no_elem 3"
echo "!!!test"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test -embedding_size 100 -pre_trained ${model} -test_only True -model GA ${option_suffix}
echo "!!!dev"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_size 100 -pre_trained ${model} -test_only True -model GA ${option_suffix}
echo "!!!test/middle"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test/middle -embedding_size 100 -pre_trained ${model} -test_only True -model GA ${option_suffix}
echo "!!!test/high"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test/high -embedding_size 100 -pre_trained ${model} -test_only True -model GA ${option_suffix}
