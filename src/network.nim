import sugar
import sequtils
import math
import strutils
import random
import system
import neo

type
  NDMatrix* = Matrix[float64]
  Vector* = Matrix[float64]
  TestData* = tuple[input, expected_result: Vector]
  Network = ref object of RootObj
    sizes : seq[int]
    biases: seq[Vector]
    weights: seq[NDMatrix]

proc `/`(x: float64, a:NDMatrix): NDMatrix =
  makeMatrix(a.dim.rows, a.dim.columns, proc(i, j: int): float64 = x / a[i, j])

proc `+`(a: NDMatrix, x: float64): NDMatrix =
  makeMatrix(a.dim.rows, a.dim.columns, proc(i, j: int): float64 = x + a[i, j])

proc `-`(x: float64, a: NDMatrix): NDMatrix =
  makeMatrix(a.dim.rows, a.dim.columns, proc(i, j: int): float64 = x - a[i, j])

proc `-`(a: NDMatrix): NDMatrix =
  makeMatrix(a.dim.rows, a.dim.columns, proc(i, j: int): float64 =  -a[i, j])

proc maxIndex(a: Vector): tuple[i: int, val: float64] =
  assert(a.dim.columns == 1)
  maxIndex(a.column(0))

proc sumColumns(m: NDMatrix): NDMatrix =
  proc addColumns(i:int): float64 =
    result = 0.0
    for j in 0..<m.dim.columns:
      result += m[i, j]
  result = makeMatrix(m.dim.rows, 1, proc(i, j:int): float64 = addColumns(i))

proc normGauss: float {.inline.} = 1.46 * cos(2*PI*rand(1.0)) * sqrt(-2*log10(rand(1.0)))

proc normalRandomNDMatrix(M, N: int): NDMatrix =
  makeMatrix(M, N, proc(i, j: int): float64 = normGauss())

proc shuffle[T](x: var seq[T]) =
  for i in countdown(x.high, 0):
    let j = rand(i + 1)
    swap(x[i], x[j])

proc sigmoid(z: NDMatrix): NDMatrix =
  result = 1.0 / (exp(-z) + 1.0)

proc sigmoid_prime(z: NDMatrix): NDMatrix =
  sigmoid(z) |*| (1.0 - sigmoid(z))

proc make_network*(sizes: seq[int]): Network =
  let biases = collect(newSeq):
      for i in (1..<sizes.len): normalRandomNDMatrix(sizes[i], 1)
  let weights = collect(newSeq):
      for i in (1..<sizes.len): normalRandomNDMatrix(sizes[i], sizes[i-1]) 
  result = Network(sizes:sizes,
                   biases:biases,
                   weights:weights)

proc feed_forward*(network: Network, a: Vector): Vector =
  assert(a.dim.columns == 1)
  result = a
  for w, b in zip(network.weights, network.biases).items:
    result = sigmoid(w * result + b)

proc cost_derivative(network: Network, output_activations, y: Vector): Vector =
  output_activations - y

proc backprop_matrix(network: Network, x, y: NDMatrix): auto =
  var
    nabla_b = collect(newSeq):
        for b in network.biases: zeros(b.dim.rows, b.dim.columns)
    nabla_w = collect(newSeq):
        for w in network.weights: zeros(w.dim.rows, w.dim.columns)
    activation = x
    activations = @[x]
    zs: seq[NDMatrix] = @[]

  for w, b in zip(network.weights, network.biases).items:
    let bias = makeMatrix(w.dim.rows, activation.dim.columns, proc(i, j:int): float64 = b[i, 0])
    let z = w * activation + bias
    zs.add(z)
    activation = sigmoid(z)
    activations.add(activation)

  var delta = network.cost_derivative(activations[^1], y) |*| sigmoid_prime(zs[^1])

  nabla_b[^1] = sumColumns(delta)
  nabla_w[^1] = delta * activations[^2].t
  for i in 2..<network.sizes.len:
    let z = zs[^i]
    let sp = sigmoid_prime(z)
    delta = (network.weights[^(i - 1)].t * delta) |*| sp
    nabla_b[^i] = sumColumns(delta)
    nabla_w[^i] = delta * activations[^(i + 1)].t
  result = (nabla_b, nabla_w)

proc update_mini_batch(network: Network, mini_batch: seq[TestData], eta: float64) =
  let inputs = makeMatrix(mini_batch[0].input.dim.rows, mini_batch.len,
                 proc(i, j: int): float64 = mini_batch[j].input[i, 0])
  let expected_results = makeMatrix(mini_batch[0].expected_result.dim.rows, mini_batch.len,
                 proc(i, j: int): float64 = mini_batch[j].expected_result[i, 0])

  let (nabla_b, nabla_w) = network.backprop_matrix(inputs, expected_results)

  for i in 0..network.biases.high:
    network.weights[i] -= (eta / toFloat(mini_batch.len)) * nabla_w[i]
    network.biases[i] -= (eta / toFloat(mini_batch.len)) * nabla_b[i]

proc evaluate*(network: Network, test_data: seq[TestData]): int =
  let test_results = collect(newSeq):
    for dat in test_data: (maxIndex(network.feed_forward(dat.input)).i, maxIndex(dat.expected_result).i)
  result = len(test_results.filter(proc(r: tuple[res, expected: int]): bool = (r.res == r.expected)))

proc sgd*(network: Network,
          training_data: seq[TestData],
          epochs: int,
          mini_batch_size: int,
          eta: float64,
          test_data: seq[TestData] = @[]) =
  var training_data_var = training_data

  for j in 0..<epochs:
    shuffle(training_data_var)
    let mini_batches = collect(newSeq):
        for k in countup(0, training_data.len - 1, mini_batch_size): toSeq(training_data_var[k..<(min(k + mini_batch_size, training_data_var.len))])
    for mini_batch in mini_batches:
      network.update_mini_batch(mini_batch, eta)
    if test_data.len > 0:
      echo("Epoch $#: $# / $#" % [$(j), $(network.evaluate(test_data)), $(test_data.len)])
    else:
      echo("Epoch $# complete" % $(j))
