# Hyperparameters search results

Random + manual search\
Best hyperparameters = best f1 score on validation set

## CharCNN

Validation results:
  * accuracy: 0.7680
  * f1: 0.7712

Hyperparameters:

  * lr: 0.001
  * maxlen: 512
  * n_filters: 128
  * cnn_kernel_size: 5
  * dropout: 0.5

  * epochs: 30

Run name:
`Jul09_14-12-47_lyalin_CharCNN_lr3_dropout0.5_noise_level0.0000hyperparameters_search`

## FastText-RNN

Validation results:
  * accuracy: f1: 0.8744
  * f1: 0.8745

Hyperparameters:
  * hidden_dim: 256
  * dropout: 0.20
  * input_dim: 300
  * lr: 0.0006

  * epochs: 20

Run name:
`Jul12_02-02-09_lyalin_RNNBinaryClassifier_lr3_dropout0.19682083244247645_noise_level0.0000hyperparameters_search_random`

## YoonKim

Validation results:
  * accuracy: 0.8475
  * f1: 0.8474

Hyperparameters:
  * n_filters: 32
  * cnn_kernel_size: 5
  * hidden_dim_out: 64
  * embedding_dim: 90
  * dropout: 0.5
  * lr: 0.001

  * epochs: 20

Run name:
  `Jul10_12-36-20_lyalin_YoonKimModel_lr3_dropout0.5_noise_level0.0000hyperparameters_search_manual`

## AttentionedYoonKim

Validation results:
  * accuracy: 0.8757
  * f1: 0.8757

Hyperparameters:

  * lr: 0.001
  * maxlen: 512
  * n_filters: 128
  * cnn_kernel_size: 5
  * dropout: 0.5
  * hidden_dim_out: 128
  * embedding_dim: 74
  
  * epochs: 20

Run name:
`Jul09_15-21-51_lyalin_AttentionedYoonKimModel_lr3_dropout0.5_noise_level0.0000hyperparameters_search`
