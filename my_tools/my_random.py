import random
import math

def randn(rows, cols):
    def single_randn():
        u1 = random.random()
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return [[single_randn() for _ in range(cols)] for _ in range(rows)]

def randn_vec(rows):
    def single_randn():
        u1 = random.random()
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return [[single_randn()] for _ in range(rows)]
