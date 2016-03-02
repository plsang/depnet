from random import shuffle, seed

seed(123)

a = [ 1, 3, 5, 7, 9]
b = range(len(a))
shuffle(b)
print(b)

