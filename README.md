Quick soduko solver written one day before christmas (2021).

Possible values for a given cell are represented as a bitmask.
Several typical moves are tried (of increasing complexity).

Usually those are enough to solve moderate to difficult puzzles.
For diabolical difficulty, do a search: pick cells who have
multiple options, and see if setting to one of those values and
iteratively applying standard moves brings to as solution. The
search is not done at a depth of more than 1 (but could be).
