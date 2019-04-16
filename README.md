# Common Modules of Neural Networks

> Collections of neural network modules and functions implemented by TensorFlow and some machine learning methods,
each method is defined in an individual `.py` file.

## Implementation List
### Neural Network Modules and Functions (Deep Learning related)
- [x] [Character-Aware Neural Language Models](http://arxiv.org/abs/1508.06615v4) for character representations, 
in [char_cnn.py](/nns/char_cnn.py).
- [x] [Highway Network](http://arxiv.org/abs/1505.00387), in [highway_network.py](/nns/highway_network.py).
- [x] [Neural Tensor Network](https://cs.stanford.edu/~danqi/papers/nips2013.pdf), in [neural_tensor_network.py](
/nns/neural_tensor_network.py).
- [x] [Self-attention Mechanism](https://arxiv.org/pdf/1409.0473.pdf), in [self_attention.py](/nns/self_attention.py).
- [ ] Gated CNN for [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083), in 
[gated_cnn.py](/nns/gated_cnn.py).
- [x] [Densely connected LSTM](https://arxiv.org/pdf/1802.00889.pdf), in [densely_connect_rnn.py](
/nns/densely_connect_rnn.py).
- [x] [Focal loss](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf), 
in [focal_loss.py](/nns/focal_loss.py).
- [ ] [Multihead attention](), in [multihead_attention.py](/nns/multihead_attention.py).
- [ ] [Transformer](), in [transformer.py](/nns/transformer.py).
- [x] [Gaussian Error Linear Unit](https://arxiv.org/abs/1606.08415), in [gelu.py](/nns/gelu.py).

### Machine Learning Methods
- [x] Kmeans clustering, in [kmeans_clustering.py](/mls/kmeans_clustering.py).
- [x] Minibatch kmeans clustering, in [kmeans_clustering_minibatch.py](/mls/kmeans_clustering_minibatch.py).
- [x] TSNE visualization, in [tsne.py](/mls/tsne.py).

### Metrics
- [x] CoNLLeval (sequence labeling metrics), in [CoNLLeval.py](/metrics/CoNLLeval.py).
- [x] BLEU, in [BLEU.py](/metrics/BLEU.py).
- [x] Rouge, in [Rouge.py](/metrics/Rouge.py).
