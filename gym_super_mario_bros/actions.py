"""Static action sets for binary to discrete action space wrappers."""

# go go go
GO_GO = [
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]

# go go go
GO_GO_GO = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]

# actions for the simple run right environment
RIGHT_ONLY = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]


# actions for very simple movement: 7
SIMPLE_MOVEMENT = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]


# actions for more complex movement: 12
COMPLEX_MOVEMENT = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]
