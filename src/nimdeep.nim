import random
import mnist_loader
import network

#import nimprof


randomize()
let train_data = mnist_load("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte")
let training_data = train_data[0..<50000]
let validation_data = train_data[50000..train_data.high()]
let test_data = mnist_load("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte")

var net = make_network(@[784, 30, 10])

net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
