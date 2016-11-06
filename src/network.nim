import future
import sequtils
import math
import strutils
import random
import system
import linalg

type Matrix* = DMatrix64
type Vector* = DMatrix64

type Network = ref object of RootObj
  sizes : seq[int]
  biases: seq[Vector]
  weights: seq[Matrix]

type TestData* = tuple[input, expected_result: Vector]

proc `/`(x: float64, a:Matrix): Matrix =
  makeMatrix(a.dim.rows, a.dim.columns, proc(i, j: int): float64 = x / a[i, j])

proc hadamard(a,b: Matrix): Matrix =
  assert(a.dim == b.dim)
  makeMatrix(a.dim.rows, a.dim.columns, proc(i, j: int): float64 = a[i, j] * b[i, j])

proc `+`(a: Matrix, x: float64): Matrix =
  makeMatrix(a.dim.rows, a.dim.columns, proc(i,j: int): float64 = x + a[i, j])

proc `-`(x: float64, a: Matrix): Matrix =
  makeMatrix(a.dim.rows, a.dim.columns, proc(i,j: int): float64 = x - a[i, j])

proc maxIndex(a: Vector): tuple[i: int, val: float64] =
  assert(a.dim.columns == 1)
  maxIndex(a.column(0))

proc normGauss: float {.inline.} = 1.46 * cos(2*PI*random(1.0)) * sqrt(-2*log10(random(1.0)))
proc normalRandomMatrix(M, N: int): DMatrix64 =
  makeMatrix(M, N, proc(i, j: int): float64 = normGauss())

proc shuffle[T](x: var seq[T]) =
  for i in countdown(x.high, 0):
    let j = random(i + 1)
    swap(x[i], x[j])

proc sigmoid(z: Matrix): Matrix =
  result = 1.0 / (exp(z * -1.0) + 1.0)

proc sigmoid_prime(z: Matrix): Matrix =
  sigmoid(z).hadamard(1.0 - sigmoid(z))

proc make_network*(sizes: seq[int]): Network =
  new(result)
  result.sizes = sizes
  result.biases = lc[normalRandomMatrix(sizes[i], 1) | (i <- 1..<len(sizes)), Vector]
  result.weights = lc[normalRandomMatrix(sizes[i], sizes[i-1]) | (i <- 1..<len(sizes)), Matrix]

proc feed_forward*(network: Network, a: Vector): Vector =
  assert(a.dim.columns == 1)
  var var_a = a
  for layer in zip(network.weights, network.biases):
    var_a = sigmoid(layer[0] * var_a + layer[1])
  result = var_a

proc cost_derivative(network: Network, output_activations, y: Vector): Vector =
  (output_activations - y)

proc backprop(network: Network, x, y: Vector): auto =
  var nabla_b = lc[zeros(b.dim.rows, b.dim.columns) | (b <- network.biases), Vector]
  var nabla_w = lc[zeros(w.dim.rows, w.dim.columns) | (w <- network.weights), Matrix]
  var activation = x
  var activations = @[x]
  var zs: seq[Matrix] = @[]

  for layer in zip(network.weights, network.biases):
    let z = layer[0] * activation + layer[1]
    zs.add(z)
    activation = sigmoid(z)
    activations.add(activation)

  var delta = network.cost_derivative(activations[activations.high], y).hadamard(sigmoid_prime(zs[zs.high]))
  nabla_b[nabla_b.high] = delta
  nabla_w[nabla_w.high] = delta * activations[activations.len() - 2].t
  for layer in 2..<network.sizes.len():
    let z = zs[zs.len() - layer]
    let sp = sigmoid_prime(z)
    delta = (network.weights[network.weights.len() - layer + 1].t * delta).hadamard(sp)
    nabla_b[nabla_b.len() - layer] = delta
    nabla_w[nabla_w.len() - layer] = delta * activations[activations.len() - layer - 1].t
  result = (nabla_b, nabla_w)

proc update_mini_batch(network: Network, mini_batch: seq[TestData], eta: float64) =
  var nabla_b = lc[zeros(b.dim.rows, b.dim.columns) | (b <- network.biases), Vector]
  var nabla_w = lc[zeros(w.dim.rows, w.dim.columns) | (w <- network.weights), Matrix]

  for dat in mini_batch:
    let (delta_nabla_b, delta_nabla_w) = network.backprop(dat.input, dat.expected_result)
    nabla_b = lc[nb.a + nb.b | (nb <- zip(nabla_b, delta_nabla_b)), Vector]
    nabla_w = lc[nw.a + nw.b | (nw <- zip(nabla_w, delta_nabla_w)), Matrix]
  network.biases = lc[b.a - (eta / toFloat(len(mini_batch))) * b.b |
                      (b <- zip(network.biases, nabla_b)), Vector]
  network.weights = lc[w.a - (eta / toFloat(len(mini_batch))) * w.b |
                      (w <- zip(network.weights, nabla_w)), Matrix]

proc evaluate*(network: Network, test_data: seq[TestData]): int =
  let test_results = lc[(maxIndex(network.feed_forward(dat.input)).i, maxIndex(dat.expected_result).i) |
                        (dat <- test_data), tuple[res, expected: int]]
  result = len(test_results.filter(proc(r: tuple[res, expected: int]): bool = (r.res == r.expected)))

proc sgd*(network: Network,
          training_data: seq[TestData],
          epochs: int,
          mini_batch_size: int,
          eta: float64,
          test_data: seq[TestData] = @[]) =
  let n_test = len(test_data)
  let n = len(training_data)
  var training_data_var = training_data

  for j in 0..<epochs:
    shuffle(training_data_var)
    let mini_batches = lc[toSeq(training_data_var[k..<(min(k + mini_batch_size, training_data_var.len()))]) |
                          (k <- countup(0, n - 1, mini_batch_size)),
                          seq[TestData]]
    for mini_batch in mini_batches:
      network.update_mini_batch(mini_batch, eta)
    if n_test > 0:
      echo("Epoch $#: $# / $#" % [$(j), $(network.evaluate(test_data)), $(n_test)])
      #echo network
    else:
      echo("Epoch $# complete" % $(j))
