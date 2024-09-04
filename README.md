# AI models

**0_Paradigm ML and DL with the NN one neuron:**\
the difference between traditional programming and machine learning, how to use dense layers, loss functions, optimizers, and model.fit

**1_Computer Vision(CV) with the MNIST Dataset and Callbacks:**\
the Fashion MNIST dataset, how to use 10 output neurons in the neural network, ReLU function, callbacks parameter, and how to split data into training and test sets

**2_Convolutional Neural Networks(CNN) in CV:**\
how convolutions improve image recognition and impact training in our deep neural network, also how the pooling technique is applied to images, convolution filters, for example, the function Conv1D(), and max pooling

**2__Basic convolution on a 2D grayscale image:**\
how convolution filters and max pooling work with a photo example

**3_CV with Real-world Images(horse-or-human):**\
how to use Image Generator to label images, normalize the image, understand its parameters, how it helps remove some convolutions to handle smaller images, and what overfitting is

**4_CNN_in_CV with Real-world Images(Cats vs Dogs Dataset):**\
to do this, use the famous Kaggle Dogs v Cats dataset and create a convolutional neural network in Tensorflow and use Keras image preprocessing utilities such as flow_from_directory on ImageDataGenerator. It also discusses how to access the accuracy and loss values, showing how these parameters change during model training, which resulted in the discovery of model overfitting.

**5_CNN Tackle Overfitting with Data Augmentation(Cats vs Dogs Dataset):**\
how to avoid overfitting with augmentation, how to use image augmentation with imageDataGenereato use image augmentation, how image augmentation helps to solve overfitting problem, effectively simulates having a larger dataset to train with image augmentation.

**6_Transfer Learning(Cats vs Dogs Dataset):**\
sefulness of transfer learning, use of the dropout parameter, how you can change the number of model classes, block or freeze the layer from overfitting, avoid overfitting with dropouts.

**6__Transfer Learning(horse-or-human):**\
similar "6_Transfer Learning(Cats vs Dogs Dataset) but using "horse-or-human" dataset.

**7_Multi-class Classification(images of hands of the english alphabet):**\
using the Sign Language MNIST dataset, which contains 28x28 images of hands depicting the 26 letters of the english alphabet, a Convolution, Deep Neurol Network(DNN) for Fashion MNIST have 10 output neurons.

**8_Tokenizing the BBC News archive(Kaggle):**\
how object is used, sentence tokenization method, sentence list encoding method, specifying token for unknown words, padding to the sequence.

**8__Tokenizing the Sarcasm Dataset(Kaggle News Headlines):**\
similar "8_Tokenizing the BBC News archive(Kaggle)".

**9_NLP the Sarcasm Dataset(Kaggle News Headlines):**\
build a train a model on a binary classifier with the the News Headlines Dataset for Sarcasm Detection Dataset, model using functions Embedding(), GlobalAveragePooling1D() in the context of natural language processing (NLP)

**9__NLP Subword Tokenization with the IMDB Reviews Dataset:**\
similar "9___NLP the Sarcasm Dataset(Kaggle News Headlines)" but model using subword text encoding with the IMDB Reviews Dataset 

**9___NLP the BBC News archive(Kaggle):**\
similar "9___NLP the Sarcasm Dataset(Kaggle News Headlines)" but model using the BBC News Classification Dataset 

**10___NLP Models with_GRU_LSTM_Conv1D IMDB Reviews Dataset:**\
Large Movie Review Dataset IMDB with models Keras' layers Flatten(), LSTM(long-short-term-memory), GRU(Gated Recurrent Unit), Conv1D(onvolutional neural networks (CNNs) for one-dimensional input data),how sequence make order the words for their impact on the meaning of the sentence, Recurrent Neural Networks help understand the impact of sequence on meaning, LSTM help understand meaning words, Bidirectional () keras layer type allows LSTMs to look forward and backward in a sentence.

**10__NLP Models with_LSTM_Conv1D Sarcasm_Dataset(Kaggle_News_Headlines):**\
similar "10___NLP Models with_GRU_LSTM_Conv1D IMDB Reviews Dataset" but using the Sarcasm Dataset(Kaggle News Headlines) with  models Keras' layers LSTM, Conv1D()

**10_NLP avoid Overfitting(Sentiment140 dataset):**\
similar "10___NLP Models with_GRU_LSTM_Conv1D IMDB Reviews Dataset" but using the Sentiment140 dataset with  models Keras' layers Conv1D()

