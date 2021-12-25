"""
An attempt to form a system of equations with constraints to solve
a portion of the problem.

Doesn't work.
"""
import numpy.linalg
from scipy import optimize


def solve_linear(puzzle):
  v = []
  m = {}
  for i in range(0, 9):
    for j in range(0, 9):
      if count_bits(puzzle.bitmap[i, j]) != 1:
        m[(i, j)] = len(v)
        l = -1
        x = 0
        options = []
        for k in range(0, 9):
          if puzzle.bitmap[i, j] & (1 << k):
            if l < 0:
              l = k + 1
            x = k + 1
            options.append(k + 1)
        v.append((l, x, options))
  A = np.zeros((27, len(v)))
  b = np.ones((27)) * 45
  row = 0
  for i in range(0, 9):
    for j in range(0, 9):
      if count_bits(puzzle.bitmap[i, j]) == 1:
        b[row + i] -= bit_to_integer(puzzle.bitmap[i, j])
      else:
        A[row + i, m[(i, j)]] = 1
  row = 9
  for j in range(0, 9):
    for i in range(0, 9):
      if count_bits(puzzle.bitmap[i, j]) == 1:
        b[row + j] -= bit_to_integer(puzzle.bitmap[i, j])
      else:
        A[row + j, m[(i, j)]] = 1
  row = 18
  for block in range(0, 9):
    block_i = block // 3
    block_j = block % 3
    for k in range(0, 9):
      i = 3 * block_i + k // 3
      j = 3 * block_j + k % 3
      if count_bits(puzzle.bitmap[i, j]) == 1:
        b[row + block] -= bit_to_integer(puzzle.bitmap[i, j])
      else:
        A[row + block, m[(i, j)]] = 1

  bounds = ([l[0] for l in v], [l[1] for l in v])
  res = optimize.lsq_linear(A, b, bounds=bounds)
  print('linear:')
  print('rank:', np.linalg.matrix_rank(A))
  print(A)
  print(bounds)
  x = res.x
  print(res)
  x_int = np.zeros((len(v)))
  for i in range(len(v)):
    print(x[i], v[i][2])
    diff = np.power(x[i] - np.array(v[i][2]), 2)
    print(diff)
    x_int[i] = v[i][2][np.argmin(diff)]
  print(x, x_int)

  print(np.dot(A, x_int) - b)
