nimdeep
=======

An experimental rewrite of the example code from http://neuralnetworksanddeeplearning.com/ in Nim.
The original Python code can be found in https://github.com/mnielsen/neural-networks-and-deep-learning.git

Installation and Running
------------------------
 - nimble install linalg
 - download and uncompress the MNIST data files from http://yann.lecun.com/exdb/mnist/ into the data directory
 - cd src
 - nim c --clib:/usr/local/lib/liblapack.so --clib:/usr/local/lib/libblas.so -d:release -d:openblas nimdeep.nim
 - ./nimdeep
