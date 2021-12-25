""" Solve a soduko puzzle.

Possible values for a given cell are represented as a bitmask.
Several typical moves are tried (of increasing complexity).

Usually those are enough to solve moderate to difficult puzzles.
For diabolical difficulty, we do a search: pick cells who have
multiple options, and see if setting to one of those values and
iteratively applying standard moves brings to as solution. The
search is not done at a depth of more than 1 (but could be).
"""
import argparse
import os
import sys
import numpy as np
import pickle

verbose = False
bit_lookup = None


def row_iterator(i, j):
  for jj in range(0, 9):
    yield i, jj


def col_iterator(i, j):
  for ii in range(0, 9):
    yield ii, j


def grid_iterator():
  for i in range(0, 9):
    for j in range(0, 9):
      yield i, j


def block_iterator(i, j):
  block_i, block_j = i // 3, j // 3
  for k in range(0, 9):
    yield 3 * block_i + k // 3, 3 * block_j + k % 3


def concat_iterators(its):
  for it in its:
    for i, j in it:
      yield i, j


def all_iterators(i, j):
  return concat_iterators(
      [row_iterator(i, j),
       col_iterator(i, j),
       block_iterator(i, j)])


def make_bitmap(state):
  bitmap = np.zeros((9, 9), 'int32')
  for i, j in grid_iterator():
    bitmap[i, j] = 0 if state[i][j] == 0 else 1 << (state[i][j] - 1)
  return bitmap


def count_bits(bitmask):

  def _count_bits(mask):
    count = 0
    for i in range(0, 9):
      if (mask & (1 << i)):
        count += 1
    return count

  global bit_lookup
  if not bit_lookup:
    bit_lookup = [_count_bits(i) for i in range(0, 512)]
  return bit_lookup[bitmask]


def bit_to_integer(bitmask):
  bit_index = 0
  for i in range(0, 9):
    if bitmask == (1 << i):
      return i + 1
  return 0


def bit_to_readable(bitmask):
  if count_bits(bitmask) != 1:
    return ' '
  bit_index = 0
  for i in range(0, 9):
    if bitmask == (1 << i):
      bit_index = i
  return chr((bit_index + 1) + ord('0'))


def bit_to_options(bitmask):
  s = ''
  for i in range(0, 9):
    if bitmask & (1 << i):
      s += chr((i + 1) + ord('0'))
    else:
      s += '-'
  return s


def choose_n(items, n, start=0):
  for k in range(start, len(items) - (n - 1)):
    if n > 1:
      for other in choose_n(items, n - 1, k + 1):
        yield tuple([k] + list(other))  # This is a bit inefficient
    else:
      yield (k,)


class Puzzle:

  def __init__(self, state):
    self.initial_state = state
    self.bitmap = make_bitmap(state)

  def __str__(self):
    s = '\n'.join([
        ' '.join([bit_to_readable(item) for item in row]) for row in self.bitmap
    ])
    options_str = '\n'.join([
        ' '.join([bit_to_options(item) for item in row]) for row in self.bitmap
    ])
    return 'num_unknown = %d, valid = %d, progress = %d\n' % (self.num_unknown(
    ), self.check_state(), self.progress()) + s + '\n' + options_str

  def num_unknown(self):
    return sum([count_bits(self.bitmap[i, j]) != 1 for i, j in grid_iterator()],
               0)

  def progress(self):
    return sum([count_bits(self.bitmap[i, j]) for i, j in grid_iterator()], -81)

  def update_entry(self, i, j):
    bits = 0x1ff if (self.bitmap[i, j] == 0) else self.bitmap[i, j]
    for ii, jj in all_iterators(i, j):
      if count_bits(self.bitmap[ii, jj]) == 1:
        bits &= ~self.bitmap[ii, jj]
    self.bitmap[i, j] = bits

  def check_state(self, verbose=False):
    messages = []
    for i in range(0, 9):
      its = {
          'row': row_iterator(i, 0),
          'col': col_iterator(0, i),
          'block': block_iterator(3 * (i // 3), 3 * (i % 3))
      }
      for it_name, it in its.items():
        bits = 0
        for ii, jj in it:
          bits |= self.bitmap[ii, jj]
        if count_bits(bits) != 9:
          messages.append('Invalid %s %d' % (it_name, i))
    if verbose:
      print(messages)
    return len(messages) == 0

  def first_pass(self):
    """Upon loading the puzzle, some cells are "0". Update them."""
    found = []
    for i, j in grid_iterator():
      if self.bitmap[i, j] == 0:
        self.update_entry(i, j)
        if count_bits(self.bitmap[i, j]) == 1:
          found.append((i, j))
    self.propagate_updates(found)

  def _apply_mask(self, i, j, mask):
    """Applies a bitmask to the given cell if it is not frozen."""
    if count_bits(self.bitmap[i, j]) != 1:
      self.bitmap[i, j] &= mask
      if count_bits(self.bitmap[i, j]) == 1:
        self.propagate_updates([(i, j)])
      return True
    return False

  def propagate_updates(self, found):
    """
    Once a cell has been frozen to a certain value, we can
    update blocks, rows, and columns to remove that value.
    """
    while len(found) > 0:
      new_found = []
      for entry in found:
        if count_bits(self.bitmap[entry[0], entry[1]]) != 1:
          continue
        bits = self.bitmap[entry[0], entry[1]]
        for ij in all_iterators(entry[0], entry[1]):
          if ij[0] == entry[0] and ij[1] == entry[1]:
            continue
          self._apply_mask(ij[0], ij[1], ~bits)

      if verbose:
        print('found/new_found %d %d\n' % (len(found), len(new_found)))
      found = new_found

  def blockwise_line(self, index, it_gen):
    """
        For each line (either row or column), if all the possible positions
        for a given value lie within a single block, then those values can
        not appear at any other place within the block (otherwise we wouldn't
        be able to put them on the given line).
        """
    for k in range(0, 9):
      unknown_by_block = [0, 0, 0]
      for ij in it_gen(k):
        i, j = ij[0], ij[1]
        if count_bits(self.bitmap[i, j]) != 1:
          unknown_by_block[ij[1 - index] // 3] |= self.bitmap[i, j]

      for b1 in range(0, 3):
        other_blocks = unknown_by_block[(b1 + 1) %
                                        3] | unknown_by_block[(b1 + 2) % 3]
        bits_to_clear = (other_blocks
                         ^ unknown_by_block[b1]) & unknown_by_block[b1]
        if bits_to_clear:
          if verbose:
            print('Blockwise rows')
          ij = [k if index == 0 else 3 * b1, k if index == 1 else 3 * b1]
          for ij2 in block_iterator(ij[0], ij[1]):
            if ij2[index] == ij[index]:
              continue
            self._apply_mask(ij2[0], ij2[1], ~bits_to_clear)

  def blockwise_rows(self):
    return self.blockwise_line(0, lambda x: row_iterator(x, 0))

  def blockwise_cols(self):
    return self.blockwise_line(1, lambda x: col_iterator(0, x))

  def blockwise(self):
    """
        For each block, if all the possible locations for a given
        value lie on one line, then we cannot put that value
        on any that line within any other block.
        """
    for block_i in range(0, 3):
      for block_j in range(0, 3):
        unknown_by_row = [0, 0, 0]
        unknown_by_col = [0, 0, 0]
        for i, j in block_iterator(3 * block_i, 3 * block_j):
          if count_bits(self.bitmap[i, j]) != 1:
            unknown_by_row[i % 3] |= self.bitmap[i, j]
            unknown_by_col[j % 3] |= self.bitmap[i, j]
        its = {
            'row': (0, unknown_by_row, row_iterator),
            'col': (1, unknown_by_col, col_iterator),
        }
        for it_name, it_data in its.items():
          index, unknown_by_line, it = it_data[0], it_data[1], it_data[2]
          for k1 in range(0, 3):
            other_lines = unknown_by_line[(k1 + 1) %
                                          3] | unknown_by_line[(k1 + 2) % 3]
            bits_to_clear = (other_lines
                             ^ unknown_by_line[k1]) & unknown_by_line[k1]
            if bits_to_clear:
              if verbose:
                print('Found some %s (block_i=%d, block_j=%d, k1=%d): %s' %
                      (it_name, block_i, block_j, k1,
                       bit_to_options(bits_to_clear)))
              # For each other column (outside this block)
              block = [block_i, block_j]
              for ij in it(3 * block[0] + k1, 3 * block[1] + k1):
                if ij[1 - index] // 3 == block[1 - index]:
                  continue
                self._apply_mask(ij[0], ij[1], ~bits_to_clear)

    return False

  def union_n(self, n):
    for i in range(0, 9):
      its = [
          row_iterator(i, 0),
          col_iterator(0, i),
          block_iterator(3 * (i // 3), 3 * (i % 3))
      ]
      for it in its:
        items = [ij for ij in it]
        for ks in choose_n(items, n):
          union_bits = 0
          for k in ks:
            union_bits |= self.bitmap[items[k][0], items[k][1]]
          ks = frozenset(ks)
          if count_bits(union_bits) == n:
            for k in range(0, 9):
              if k in ks:
                continue
              self._apply_mask(items[k][0], items[k][1], ~union_bits)

  def only_place(self):
    for i, j in grid_iterator():
      if count_bits(self.bitmap[i, j]) == 1:
        continue
      iterators = [row_iterator, col_iterator, block_iterator]
      for it in iterators:
        other = 0
        for ii, jj in it(i, j):
          if ii == i and jj == j:
            continue
          other |= self.bitmap[i, jj]
        sel = (other ^ self.bitmap[i, j]) & self.bitmap[i, j]
        if count_bits(sel) == 1:
          self.bitmap[i, j] = sel
          self.propagate_updates([(i, j)])
          break

  def search(self, max_level):
    bitmap_copy = self.bitmap.copy()
    num_attempts = 0
    for i, j in grid_iterator():
      if count_bits(bitmap_copy[i, j]) == 2:
        print('in search, num == 2 (%d, %d)\n' % (i, j))
        num_attempts += 1
        for k in range(0, 9):
          if bitmap_copy[i, j] & (1 << k):
            self.bitmap = bitmap_copy.copy()
            self.bitmap[i, j] = (1 << k)
            self.solve(max_level)
            print('in search: %d\n' % self.progress())
            if self.progress() == 0 and self.check_state():
              print('in search, solved: %d' % num_attempts)
              return
            elif False and (self.progress() > 0 and self.check_state()):
              print(solve_linear(self))

  def solve(self, max_level):
    self.first_pass()
    print(puzzle)

    last_progress = 9**3
    # Types of moves in increasing difficulty
    moves = [
        ('only_place', 1, lambda x: x.only_place()),
        ('blockwise', 2, lambda x: x.blockwise()),
        ('blockwise_rows', 2, lambda x: x.blockwise_rows()),
        ('blockwise_cols', 2, lambda x: x.blockwise_cols()),
        ('union_2', 3, lambda x: x.union_n(2)),
        ('union_3', 4, lambda x: x.union_n(3)),
        # Never found a puzzle that needed this
        # ("union_4", 5, lambda x: x.union_n(4)),
    ]
    while self.progress() != last_progress:
      last_progress = self.progress()
      for m in moves:
        if m[1] > max_level:
          break
        m[2](self)
        if verbose:
          print(puzzle)
        if not self.check_state():
          print(m[0])
          return False
        if self.progress() == 0:
          break
    return self.progress() == 0


def read_puzzle(filename):
  with open(filename, 'r') as f:
    lines = f.readlines()
    state = [[ord(c[0]) - ord('0') for c in line.split(' ')] for line in lines]
  return Puzzle(state)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Solve sodu.')
  parser.add_argument(
      '--max_level',
      type=int,
      default=5,
      help='Max level (complexity of moves)')
  parser.add_argument('--puzzle', help='Filename of the puzzle')
  parser.add_argument(
      '--allow_search', type=int, help='Allow performing search', default=1)
  args = parser.parse_args(sys.argv[1:])
  puzzle = read_puzzle(args.puzzle)

  puzzle.solve(args.max_level)
  print(puzzle)
  if args.allow_search and (puzzle.progress() != 0):
    print('searcing ---------------------------------------')
    puzzle.search(args.max_level)
    print(puzzle)
  if puzzle.progress() == 0:
    sys.exit(0)

  sys.exit(1)
