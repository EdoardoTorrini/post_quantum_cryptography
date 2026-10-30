import secrets, random

import numpy as np


def poly_mul_ring(p1: np.array, p2: np.ndarray, q: int, N: int) -> np.ndarray:
    p = np.convolve(p1, p2)
    ret, tail = p[:N].copy(), p[N:]
    ret[:len(tail)] -= tail
    return ret % q


def symmetric_mods(r: int, p: int) -> int:
    r = r % p
    if r <= (p - 1) // 2:
        return r
    return r - p 


def get_param() -> tuple:
    q = 3329
    n = 256
    k = 3
    eta_1, eta_2 = 2, 2
    return (q, n, k, eta_1, eta_2)


def sample_cbd(n: int, eta: int) -> np.ndarray:
    a = np.random.randint(0, 2, size=(n, eta)).sum(axis=1)
    b = np.random.randint(0, 2, size=(n, eta)).sum(axis=1)
    return a - b


def key_gen() -> tuple:
    q, n, k, eta_1, _ = get_param()
    A = [ [ np.random.randint(0, q, size=n) for _ in range(k) ] for _ in range(k) ]
    
    # for construct the vector s and e we have to took a random small value centerd in 0
    # CBD: centered binomial distribution
    # a_i = [-eta_1, ..., 0, ..., eta_1]

    s = [ sample_cbd(n, eta_1) for _ in range(k) ]
    e = [ sample_cbd(n, eta_1) for _ in range(k) ]

    t = []
    for i in range(k):
        t_i = np.zeros(n, dtype=int)
        for j in range(k):
            t_i = ( t_i + poly_mul_ring(A[i][j], s[j], q, n) ) % q
        t_i = (t_i + e[i]) % q
        t.append(t_i)

    return (A, t), s


def encrypt(m: bytes, pk: tuple) -> tuple:
    q, n, k, eta_1, eta_2 = get_param()
    A, t = pk
    m_bits = np.unpackbits(np.frombuffer(m, dtype=np.uint8)).astype(int)
    
    r = [ sample_cbd(n, eta_1) for _ in range(k) ]
    e_1 = [ sample_cbd(n, eta_2) for _ in range(k) ]
    e_2 = sample_cbd(n, eta_2)

    u = []
    for i in range(k):
        u_i = np.zeros(n, dtype=int)
        for j in range(k):
            u_i = ( u_i + poly_mul_ring(A[j][i], r[j], q, n) ) % q
        u_i = (u_i + e_1[i]) % q
        u.append(u_i)
    
    m_scaled = m_bits * ( (q + 1) // 2 )

    v = np.zeros(n, dtype=int)
    for j in range(k):
        v = ( v + poly_mul_ring(t[j], r[j], q, n) ) % q

    v = (v + e_2 + m_scaled) % q
    return (u, v)


def decrypt(c: tuple, sk: list) -> bytes:
    q, n, k, _, _ = get_param()
    u, v = c

    su = np.zeros(n, dtype=int)
    for j in range(k):
        su = ( su + poly_mul_ring(sk[j], u[j], q, n) ) % q

    noisy_bit = ( v - su ) % q

    # round function m = Round_q(v - s^T * u)
    m_bits = []
    for val in noisy_bit:

        val_c = symmetric_mods(val, q)
        if abs(val_c) > q // 4:
            m_bits.append(1)
        else:
            m_bits.append(0)

    return np.packbits(m_bits).tobytes()


def main():
    pk, sk = key_gen()

    # shared secret used for symmetric cryptography
    m = secrets.randbits(256).to_bytes(32, "big")
    print(f"{m = }")
    
    u, v = encrypt(m, pk)
    print(f"len u: {len(u)}, len v: {len(v)}")

    m_sol = decrypt((u, v), sk)
    print(f"{m_sol = }")

    assert m == m_sol

if __name__ == "__main__":
    main()
