nimdeep
=======

An experimental rewrite of the example code from http://neuralnetworksanddeeplearning.com/ in Nim.
The original Python code can be found in https://github.com/mnielsen/neural-networks-and-deep-learning.git

Installation and Running
------------------------
 - nimble install linalg
 - cd src
 - nim c nim c --clib:/usr/local/lib/liblapack.so --clib:/usr/local/lib/libblas.so -d:release nimdeep.nim
 - ./nimdeep
