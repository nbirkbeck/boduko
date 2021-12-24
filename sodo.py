import os
import sys
import numpy as np
import pickle

verbose = False
bit_lookup = None

def make_bitmap(state):
    bitmap = np.zeros((9, 9), 'int32')
    for i in range(0, 9):
        for j in range(0, 9):
            if state[i][j] == 0:
                bitmap[i, j] = 0
            else:
                bitmap[i, j] = 1 << (state[i][j] - 1)
    return bitmap

def count_bits(bitmask):
    if bit_lookup:
        return bit_lookup[bitmask]
    count = 0
    for i in range(0, 9):
        if (bitmask & (1 << i)):
            count += 1
    return count

def make_table():
    return [count_bits(i) for i in range(0, 512)]

bit_lookup = make_table()


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
    return concat_iterators([row_iterator(i, j),
                             col_iterator(i, j),
                             block_iterator(i, j)])

class Puzzle:
    def __init__(self, state):
        self.initial_state = state
        self.bitmap = make_bitmap(state)

    def __str__(self):
        puzzle = self.initial_state
        s = '\n'.join([' '.join([bit_to_readable(item) for item in row]) for row in self.bitmap])
        puzzle = self.bitmap
        hex_str = '\n'.join([' '.join(['%08x' % item for item in row]) for row in puzzle])
        options_str = '\n'.join([' '.join([bit_to_options(item) for item in row]) for row in self.bitmap])
        return 'num_unknown = %d, valid = %d, progress = %d\n' % (self.num_unknown(), self.check_state(), self.progress()) + s + '\n' + options_str

    def num_unknown(self):
        return sum([count_bits(self.bitmap[i, j]) != 1
                    for i, j in grid_iterator()], 0)

    def progress(self):
        return sum([count_bits(self.bitmap[i, j])
                    for i, j in grid_iterator()], -81)

    def update_entry(self, i, j):
        bits = 0x1ff if (self.bitmap[i, j] == 0) else self.bitmap[i, j]
        for ii,jj in all_iterators(i, j):
            if count_bits(self.bitmap[ii, jj]) == 1:
                bits &= ~self.bitmap[ii, jj]
        self.bitmap[i, j] = bits

    def check_state(self, verbose=False):
        invalid = False
        messages = []
        for i in range(0, 9):
            its = {"row": row_iterator(i, 0),
                   "col": col_iterator(0, i),
                   "block": block_iterator(3 * (i // 3), 3  * (i % 3))}
            for it_name, it in its.items():
                bits = 0
                for ii, jj in it:
                    bits |= self.bitmap[ii, jj]
                if count_bits(bits) != 9:
                    messages.append('Invalid %s %d' % (it_name, i))
        if verbose: print(messages)
        return len(messages) == 0

    def first_pass(self):
        found = []
        for i, j in grid_iterator():
            if self.bitmap[i, j] == 0:
                self.update_entry(i, j)
                if count_bits(self.bitmap[i, j]) == 1:
                    found.append((i, j))
        self.propagate_updates(found)

    def propagate_updates(self, found):
        while len(found) > 0:
            new_found = []
            for entry in found:
                if count_bits(self.bitmap[entry[0], entry[1]]) != 1:
                    continue
                bits = self.bitmap[entry[0], entry[1]]
                for ij in all_iterators(entry[0], entry[1]):
                    if ij[0] == entry[0] and ij[1] == entry[1]: continue
                    if count_bits(self.bitmap[ij[0], ij[1]]) == 1: continue
                    self.bitmap[ij[0], ij[1]] &= ~bits
                    if count_bits(self.bitmap[ij[0], ij[1]]) == 1:
                        new_found.append((ij[0], ij[1]))
            if verbose:
                print('found/new_found %d %d\n' % (len(found), len(new_found)))
            found = new_found

    def blockwise_rows(self):
        for i in range(0, 9):
            unknown_by_block = [0, 0, 0]
            for j in range(0, 9):
                if count_bits(self.bitmap[i, j]) != 1:
                    unknown_by_block[j // 3] |= self.bitmap[i, j]

            for block_j in range(0, 3):
                other_blocks = 0
                for kk in range(0, 3):
                    if block_j == kk: continue
                    other_blocks |= unknown_by_block[kk]
                bits_to_clear = (other_blocks ^ unknown_by_block[block_j]) & unknown_by_block[block_j]
                if bits_to_clear:
                    if verbose:
                        print('Blockwise rows')
                    for ii in range(0, 3):
                        I = 3 * (i // 3) + ii
                        if i == I: continue
                        for jj in range(0, 3):
                            J = 3 * block_j + jj
                            if count_bits(self.bitmap[I, J]) != 1:
                                self.bitmap[I, J] &= ~bits_to_clear
                                if count_bits(self.bitmap[I, J]) == 1:
                                    self.propagate_updates([(I, J)])
    def blockwise_cols(self):
        for j in range(0, 9):
            unknown_by_block = [0, 0, 0]
            for i in range(0, 9):
                if count_bits(self.bitmap[i, j]) != 1:
                    unknown_by_block[i // 3] |= self.bitmap[i, j]

            for block_i in range(0, 3):
                other_blocks = 0
                for kk in range(0, 3):
                    if block_i == kk: continue
                    other_blocks |= unknown_by_block[kk]
                bits_to_clear = (other_blocks ^ unknown_by_block[block_i]) & unknown_by_block[block_i]
                if bits_to_clear:
                    if verbose: print('Blockwise cols')
                    for jj in range(0, 3):
                        J = 3 * (j // 3) + jj
                        if j == J: continue
                        for ii in range(0, 3):
                            I = 3 * block_i + ii
                            if count_bits(self.bitmap[I, J]) != 1:
                                self.bitmap[I, J] &= ~bits_to_clear
                                if count_bits(self.bitmap[I, J]) == 1:
                                    self.propagate_updates([(I, J)])

    def blockwise(self):
        for block_i in range(0, 3):
            for block_j in range(0, 3):
                unknown = 0
                unknown_by_row = [0, 0, 0]
                unknown_by_col = [0, 0, 0]
                for ii in range(0, 3):
                    for jj in range(0, 3):
                        i = 3 * block_i + ii
                        j = 3 * block_j + jj
                        if count_bits(self.bitmap[i, j]) != 1:
                            unknown |= self.bitmap[i, j]
                            unknown_by_row[ii] |= self.bitmap[i, j]
                            unknown_by_col[jj] |= self.bitmap[i, j]
                for ii in range(0, 3):
                    other_rows = 0
                    for k in range(0, 3):
                        if k == ii: continue
                        other_rows |= unknown_by_row[k]
                    bits_to_clear = (other_rows ^ unknown_by_row[ii]) & unknown_by_row[ii]
                    if bits_to_clear:
                        if verbose:
                            print('Found some rows (block_i=%d, block_j=%d, ii=%d): %s' % (
                                block_i, block_j, ii, bit_to_options(bits_to_clear)))
                        # For each other column (outside this block)
                        for j in range(0, 9):
                            if j // 3 == block_j: continue
                            if count_bits(self.bitmap[3 * block_i + ii, j]) != 1:
                                self.bitmap[3 * block_i + ii, j] &= ~bits_to_clear
                                if count_bits(self.bitmap[3 * block_i + ii, j]) == 1:
                                    self.propagate_updates([(3 * block_i + ii, j)])

                for jj in range(0, 3):
                    other_cols = 0
                    for k in range(0, 3):
                        if k == jj: continue
                        other_cols |= unknown_by_col[k]
                    bits_to_clear = (other_cols ^ unknown_by_col[jj]) & unknown_by_col[jj]
                    if bits_to_clear:
                        if verbose:
                            print('Found some cols (block_i=%d, block_j=%d, jj=%d): %s' % (block_i, block_j, jj, bit_to_options(bits_to_clear)))
                        # For each row column (outside this block)
                        for i in range(0, 9):
                            if i // 3 == block_i: continue
                            if count_bits(self.bitmap[i, 3 * block_j + jj]) != 1:
                                self.bitmap[i, 3 * block_j + jj] &= ~bits_to_clear
                                if count_bits(self.bitmap[i, 3 * block_j + jj]) == 1:
                                    self.propagate_updates([(i, 3 * block_j + jj)])

        return False

    def union_2(self):
        # Row-based
        for i in range(0, 9):
            for j1 in range(0, 8):
                if count_bits(self.bitmap[i, j1]) == 1: continue
                for j2 in range(j1 + 1, 9):
                    if count_bits(self.bitmap[i, j2]) == 1: continue
                    union_bits = self.bitmap[i, j1] | self.bitmap[i, j2]
                    if count_bits(union_bits) == 2:
                        for j in range(0, 9):
                            if j == j1: continue
                            if j == j2: continue
                            if count_bits(self.bitmap[i, j]) == 1: continue
                            self.bitmap[i, j] &= ~union_bits
                            if count_bits(self.bitmap[i, j]) == 1:
                                self.propagate_updates([(i, j)])

        # Col-based
        for j in range(0, 9):
            for i1 in range(0, 8):
                if count_bits(self.bitmap[i1, j]) == 1: continue
                for i2 in range(i1 + 1, 9):
                    if count_bits(self.bitmap[i2, j]) == 1: continue

                    union_bits = self.bitmap[i1, j] | self.bitmap[i2, j]
                    if count_bits(union_bits) == 2:
                        for i in range(0, 9):
                            if i == i1: continue
                            if i == i2: continue
                            if count_bits(self.bitmap[i, j]) == 1: continue
                            self.bitmap[i, j] &= ~union_bits
                            if count_bits(self.bitmap[i, j]) == 1:
                                self.propagate_updates([(i, j)])


        # Block-based
        for block in range(0, 9):
            block_i = block // 3
            block_j = block % 3
            for k1 in range(0, 8):
                i1 = 3 * block_i + k1 // 3
                j1 = 3 * block_j + k1 % 3
                if count_bits(self.bitmap[i1, j1]) == 1: continue
                for k2 in range(k1 + 1, 9):
                    i2 = 3 * block_i + k2 // 3
                    j2 = 3 * block_j + k2 % 3
                    if count_bits(self.bitmap[i2, j2]) == 1: continue
                    union_bits = self.bitmap[i1, j1] | self.bitmap[i2, j2]
                    if count_bits(union_bits) == 2:
                        if verbose: print('block based!')
                        for k in range(0, 9):
                            i = 3 * block_i + k // 3
                            j = 3 * block_j + k % 3
                            if (i == i1) and (j == j1): continue
                            if (i == i2) and (j == j2): continue
                            if count_bits(self.bitmap[i, j]) == 1: continue
                            self.bitmap[i, j] &= ~union_bits
                            if count_bits(self.bitmap[i, j]) == 1:
                                self.propagate_updates([(i, j)])

    def union_3(self):
        # Block-based
        for block in range(0, 9):
            block_i = block // 3
            block_j = block % 3
            for k1 in range(0, 7):
                i1 = 3 * block_i + k1 // 3
                j1 = 3 * block_j + k1 % 3
                if count_bits(self.bitmap[i1, j1]) == 1: continue
                for k2 in range(k1 + 1, 8):
                    i2 = 3 * block_i + k2 // 3
                    j2 = 3 * block_j + k2 % 3
                    if count_bits(self.bitmap[i2, j2]) == 1: continue
                    for k3 in range(k2 + 1, 9):
                        i3 = 3 * block_i + k3 // 3
                        j3 = 3 * block_j + k3 % 3
                        if count_bits(self.bitmap[i3, j3]) == 1: continue
                        union_bits = self.bitmap[i1, j1] | self.bitmap[i2, j2] | self.bitmap[i3, j3]
                        if count_bits(union_bits) == 3:
                            print('union based-3: %s' % bit_to_options(union_bits))
                            for k in range(0, 9):
                                i = 3 * block_i + k // 3
                                j = 3 * block_j + k % 3
                                if (i == i1) and (j == j1): continue
                                if (i == i2) and (j == j2): continue
                                if (i == i3) and (j == j3): continue

                                if count_bits(self.bitmap[i, j]) == 1: continue
                                self.bitmap[i, j] &= ~union_bits
                                if count_bits(self.bitmap[i, j]) == 1:
                                    self.propagate_updates([(i, j)])

    def only_place(self):
        for i, j in grid_iterator():
            if count_bits(self.bitmap[i, j]) == 1: continue
            iterators = [row_iterator, col_iterator, block_iterator]
            for it in iterators:
                other = 0
                for ii, jj in it(i, j):
                    if ii == i and jj == j: continue
                    other |= self.bitmap[i, jj]
                sel = (other ^ self.bitmap[i, j]) & self.bitmap[i, j]
                if count_bits(sel) == 1:
                    self.bitmap[i, j] = sel
                    self.propagate_updates([(i, j)])
                    break

    def solve_linear(self):
        import numpy.linalg
        from scipy import optimize

        v = []
        m = {}
        for i in range(0, 9):
            for j in range(0, 9):
                if count_bits(self.bitmap[i, j]) != 1:
                    m[(i,j)] = len(v)
                    l = -1
                    x = 0
                    options = []
                    for k in range(0, 9):
                        if self.bitmap[i, j] & (1 << k):
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
                if count_bits(self.bitmap[i, j]) == 1:
                    b[row + i] -= bit_to_integer(self.bitmap[i, j])
                else:
                    A[row + i, m[(i, j)]] = 1
        row = 9
        for j in range(0, 9):
            for i in range(0, 9):
                if count_bits(self.bitmap[i, j]) == 1:
                    b[row + j] -= bit_to_integer(self.bitmap[i, j])
                else:
                    A[row + j, m[(i, j)]] = 1
        row = 18
        for block in range(0, 9):
            block_i = block // 3
            block_j = block % 3
            for k in range(0, 9):
                i = 3 * block_i + k // 3
                j = 3 * block_j + k % 3
                if count_bits(self.bitmap[i, j]) == 1:
                    b[row + block] -= bit_to_integer(self.bitmap[i, j])
                else:
                    A[row + block, m[(i, j)]] = 1

        bounds=([l[0] for l in v],
                [l[1] for l in v])
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


    def search(self, max_level):
        print('searcing ---------------------------------------')
        bitmap_copy = self.bitmap.copy()
        num_attempts = 0
        for i in range(0, 9):
            for j in range(0, 9):
                if count_bits(bitmap_copy[i, j]) == 2:
                    print('in search, num == 2 (%d, %d)\n' % ( i, j))
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
                            elif (self.progress() > 0 and self.check_state()):
                                print(self.solve_linear())


    def solve(self, max_level):
        self.first_pass()
        print(puzzle)

        last_progress = 10000
        moves = [
            ("only_place", 1, lambda x: x.only_place()),
            ("blockwise", 2, lambda x: x.blockwise()),
            ("blockwise_rows", 2, lambda x: x.blockwise_rows()),
            ("blockwise_cols", 2, lambda x: x.blockwise_cols()),
            ("union_2", 3, lambda x: x.union_2()),
            ("union_3", 4, lambda x: x.union_3()),
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
        state = [[ord(c[0]) - ord('0')
                  for c in line.split(' ')] for line in lines]
    return Puzzle(state)


puzzle = read_puzzle(sys.argv[1])

puzzle.solve(4)
print(puzzle)
if puzzle.progress() != 0:
    puzzle.search(4)
    print(puzzle)
