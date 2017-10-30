echo "Training encoder, interaction and selection modules"
THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_file ../data/embedding/glove.6B.100d.txt -optimizer sgd -dropout_rate 0.5 -lr 0.3 -num_GA_layers 1 -hidden_size 128 -model GA -model_file ../obj/model_GA_pretrained.pkl.gz -no_elem 0

echo "Training selection and elimination modules"
THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_file ../data/embedding/glove.6B.100d.txt -optimizer sgd -dropout_rate 0.5 -lr 0.3 -num_GA_layers 1 -hidden_size 128 -model GA -model_file ../obj/model_PRE.pkl.gz -no_elem 3 -pre_trained ../obj/model_GA_pretrained.pkl.gz -end_to_end False

