import sys
import math as mt
import numpy as np
from typing import List
from typeguard import typechecked


@typechecked
def inner_product(u: np.array, v: np.array) -> np.int64:
    sum = 0
    for a, b in zip(u, v):
        sum += a * b
    return sum

@typechecked
def l2_norm(u: np.array) -> np.float64:
    sum = 0
    for el in u:
        sum += pow(el, 2)
    return pow(sum, 0.5)

@typechecked
def gauss_algorithm(u: np.array, v: np.array) -> List[np.array]:
    c = 1
    while c != 0:
        mu = inner_product(u, v) / pow(l2_norm(u), 2)
        c = int(mu + mt.copysign(0.5, mu))
        v = v - c * u
        if l2_norm(u) > l2_norm(v):
            u, v = v, u
    return [u, v]


def main():
    assert len(sys.argv) == 3
    
    u = np.array([ int(el.strip()) for el in sys.argv[1].split(",") ])
    v = np.array([ int(el.strip()) for el in sys.argv[2].split(",") ])

    assert len(u) == len(v) and len(u) == 2 and len(v) == 2

    u1, v1 = gauss_algorithm(u, v)
    print(f"{u1 = }\n{v1 = }")

if __name__ == "__main__":
    main()


