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
def size_reduction(ord: int, basis: List[np.array], gs_basis: List[np.array]) -> np.array:
    for i in range(1, len(basis)):
        for j in range(i - 1, -1, -1):
            mu_ij = inner_product(basis[i], gs_basis[j]) / pow(l2_norm(gs_basis[j]), 2)
            basis[i] = basis[i] - mt.floor(mu_ij + 0.5) * basis[j]
    return basis


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


@typechecked
def lll(ord: int, basis: List[np.array]) -> List[np.array]:
    while True:
        gs_basis = gram_schmidt(ord, basis) 
        basis = size_reduction(ord, basis, gs_basis)
        for j in range(1, len(gs_basis)):
            # swap routing
            mu = inner_product(basis[j], gs_basis[j - 1]) / pow(l2_norm(gs_basis[j - 1]), 2)
            if pow(l2_norm(gs_basis[j]), 2) < (0.75 - pow(mu, 2)) * pow(l2_norm(gs_basis[j - 1]), 2):
                basis[j - 1], basis[j] = basis[j], basis[j - 1]
                break
        else:
            break
    return basis


@typechecked
def main(verbose: bool = False):
    ret = gauss_algorithm(np.array([3, 5]), np.array([4, 7]))
    print(ret)

    ret = gram_schmidt(4, [ np.array([ -3, -1, 0, 3 ]), np.array([ 3, 2, -1, -9 ]) ])
    print(ret)

    ret = gram_schmidt(4, [ np.array([ -6, 2, 4, 4 ]), np.array([ -7, -3, 4, -4 ]), np.array([ 6, -2, 5, 5 ]) ])
    print(ret)
    
    basis = [
        np.array([92, 44, 95, 5, 97]),
        np.array([58, 43, 99, 37, 68]),
        np.array([26, 95, 16, 89, 33]),
        np.array([17, 51, 55, 42, 82]),
        np.array([24, 89, 92, 59, 92])
    ]

    ret = lll(5, basis)
    print(ret)
    
    if not verbose:
        return

    for i, vector in enumerate(basis):
        print(f"basis len x{i} vector: {np.round(l2_norm(vector), 1)}")
    for i, vector in enumerate(ret):
        print(f"lll len x{i} vector: {np.round(l2_norm(vector), 1)}")

if __name__ == "__main__":
    main(verbose=False)


