# CNN-RS: A Convolutional Neural Network Implementation in Rust

This is an implementation of a Convolutional Neural Network in Rust, which is specifically written to process the raw [MNIST dataset](http://yann.lecun.com/exdb/mnist/), as well as other input images in the same format, and classify those images as digits in the range from 0 to 9. You can download the dataset from that link.

This repository is discussed [in this blog post](https://mikecvet.medium.com/convolutional-neural-networks-from-the-ground-up-for-the-ml-adjacent-35d530ab26f3).

This neural network is built from scratch, leveraging only the [ndarray](https://crates.io/crates/ndarray) crate for linear algebraic operations. The program can also be configured to test against a directory of custom jpeg images, formatted as 28x28 pixel grayscale squares with an inverted colors (255 is black, 0 is white). For example, I tested its accuracy against these:

![The author's handwritten digits](https://github.com/mikecvet/cnn-rs/blob/master/static/digits.png)

The `cnn-rs` binary is a simple CNN with just three layers; a convolutional, max-pooling, and softmax-based fully-connected classification layer. 

![The author's handwritten digits](https://github.com/mikecvet/cnn-rs/blob/master/static/cnn_diagram.png)

Based on empirical testing, this model can reach a 98%+ accuracy rate against the MNIST test corpus.

![Entropy loss and accuracy visualization across 1 training epoch](https://github.com/mikecvet/cnn-rs/blob/master/static/loss.png)

Information about program arguments and flags:

```
Usage: cnn-rs [OPTIONS]

Options:
      --training_images <path>  Path to MNIST training image file
      --training_labels <path>  Path to MNIST image labels file
      --test_images <path>      Path to MNIST test image file
      --test_labels <path>      Path to MNIST test labels file
      --epochs <num>            Number of epochs to run over the training dataset; defaults to 3
      --learning_rate <float>   Learning rate; defaults to 0.01
      --save                    if true, will save computed model weights to ./cnn.data, overwriting any existing local file
      --load <FILE_PATH>        path to a previously-written file containing model data
      --input_dir <FILE_PATH>   path to a directory containing 28x28 px jpeg files
      --cheat                   if true, will use the contents of files under the input_dir argument to fine-tune the model before predicting outputs
  -h, --help                    Print help
  -V, --version 
```

If you modify the code, there are quite a few tests you can use to verify behavior:

```
$ cargo test
...
running 30 tests
test approx::tests::test_equal ... ok
test approx::tests::test_not_equal ... ok
test convolution::tests::test_init_dimensions ... ok
...
test pooling::tests::test_back_propagation_output_shape ... ok
```

An example run, which:
 - trains based on the MNIST training images and label files
 - specifies two full training epochs over that training set
 - loads previously-trained model data from disk
 - overwrites the updated model data to disk
 - specifies a learning rate of 1e-6

looks like this:

```
~/code/cnn-rs ~>> ./target/release/cnn-rs --training_images ./train-images.idx3-ubyte --training_labels ./train-labels.idx1-ubyte --epochs 2 --save --load ./cnn.model --learning_rate 0.000001

loaded model data from ./cnn.model
>> epoch 1 / 2
1000 examples, avg loss 0.0881 accuracy 98.3% avg batch time 781ms
2000 examples, avg loss 0.0732 accuracy 98.6% avg batch time 790ms
3000 examples, avg loss 0.0683 accuracy 98.9% avg batch time 797ms
4000 examples, avg loss 0.0962 accuracy 98.3% avg batch time 792ms
...
60000 examples, avg loss 0.0604 accuracy 98.6% avg batch time 779ms
model data saved to cnn.model
```

