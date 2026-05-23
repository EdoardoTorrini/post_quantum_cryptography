import sys
import math as mt
import numpy as np
from typing import List
from typeguard import typechecked


@typechecked
def inner_product(u: np.array, v: np.array) -> np.float64:
    sum = 0
    for a, b in zip(u, v):
        sum += a * b
    return np.float64(sum)

@typechecked
def l2_norm(u: np.array) -> np.float64:
    sum = 0
    for el in u:
        sum += pow(el, 2)
    return np.float64(pow(sum, 0.5))

@typechecked
def gauss_algorithm(u: np.array, v: np.array) -> List[np.array]:
    c = 1
    while c != 0:
        mu = inner_product(u, v) / pow(l2_norm(u), 2)
        c = int(mu + mt.copysign(0.5, mu))
        v = v - c * u
        if l2_norm(u) > l2_norm(v):
            u, v = v, u
    return [np.array(u), np.array(v)]

@typechecked
def gram_schmidt(ord: int, basis: List[np.array]) -> List[np.array]:
    gs_basis = [ basis[0] ]
    for i, el in enumerate(basis):
        factor = 0
        for j in range(i):
            mu_ij = inner_product(basis[i], gs_basis[j]) / pow(l2_norm(gs_basis[j]), 2)
            factor += mu_ij * gs_basis[j]
        if i != 0: gs_basis.append(np.array([ np.round(el, 3) for el in basis[i] - factor  ]))
    return gs_basis 

def main():

    ret = gauss_algorithm(np.array([3, 5]), np.array([4, 7]))
    print(ret)

    ret = gram_schmidt(4, [ np.array([ -3, -1, 0, 3 ]), np.array([ 3, 2, -1, -9 ]) ])
    print(ret)

    ret = gram_schmidt(4, [ np.array([ -6, 2, 4, 4 ]), np.array([ -7, -3, 4, -4 ]), np.array([ 6, -2, 5, 5 ]) ])
    print(ret)

if __name__ == "__main__":
    main()


