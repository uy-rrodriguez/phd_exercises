# Text classification with SVM and NN

The examples here are based on the instructions found in `res/TP1.pdf`, to
predict the sentiment of Tweets using Support Vector Machine and Neuronal
Networks.


## SVM

- `svm_preprocess.py`:
   Pre-processes data to generate input files in SVM format to read with
   liblinear.
- `svm.py`: Executes the SVM with liblinear, reading the files generated with
   the previous script.


## Neural Networks

- `dnn_simple_bow.py`:
   Simplistic approach where each tweet is transformed in a vector of size N,
   each dimension representing a word in the lexicon and the value being the
   number of times each word appears in the text.
   N is the number of words in the lexicon found in the test data.
- `dnn_advanced.py`:
  This example uses more advanced functionality from PyTorch:
  - Tensors are marked to be processed by the GPU when available.
  - Usage of a layer `EmbeddingBag`, that avoids the big vectors required in the
    basic example. Multiple tweets are concatenated and then offsets are used by
    the layer to determine their start and end positions. The layer outputs
    embeddings of equal size.
