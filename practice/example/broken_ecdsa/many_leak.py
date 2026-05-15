import random, string
import pwn
import sys
import argparse
import hashlib
from fpylll import IntegerMatrix, LLL

from Crypto.PublicKey import ECC


CURVE = "P-256"

def get_G(cl: pwn.tubes.remote.connect) -> ECC.EccPoint:
    tmp = cl.recvline().decode().split("=")[1].strip()[1:-1]
    x, y = int(tmp.split(",")[0].strip(), 16), int(tmp.split(",")[1].strip(), 16)
    return ECC.EccPoint(x, y, curve=CURVE)

def get_n(cl: pwn.tubes.remote.connect) -> int:
    tmp = cl.recvline().decode().split("=")[1].strip()
    return int(tmp)

def get_pk(cl: pwn.tubes.remote.connect) -> ECC.EccPoint:
    tmp = cl.recvline().decode().split("=")[1].strip()[1:-1]
    x, y = int(tmp.split(",")[0].strip(), 16), int(tmp.split(",")[1].strip(), 16)
    return ECC.EccPoint(x, y, curve=CURVE)

def sign_msg(cl: pwn.tubes.remote.connect, m: str) -> tuple:
    cl.sendlineafter(b"sign: ", m.encode())
    tmp = cl.recvline().decode().split(":")[1].strip()[1:-1]
    r, s = int(tmp.split(",")[0].strip(), 16), int(tmp.split(",")[1].strip(), 16)
    leak = int(tmp.split(",")[2].strip(), 16)
    return r, s, leak

gen_msg = lambda x: "".join([ random.choice(string.ascii_letters) for _ in range(x) ])

def perform_attack(m: list, r: list, s: list, leak: list, n: int, nleak: int) -> tuple:
    N, ell = len(m), nleak
    K, inv_2ell = n >> ell, pow(2, -ell, n)

    mN, rN, sN, leakN = m[-1], r[-1], s[-1], leak[-1]
    t_val, u_val = [], []

    for i in range(N - 1):
        t = - (pow((s[i] * rN), -1, n) * (sN * r[i])) % n
        u_prime = pow((s[i] * rN), -1, n) * (mN * r[i] - m[i] * rN) % n
        
        u = ((leak[i] + t * leakN + u_prime) * inv_2ell) % n
        t_val.append(t)
        u_val.append(u)

    # construction of the (N + 1)x(N + 1) matrix
    dim = N + 1
    B = IntegerMatrix(dim, dim)

    for i in range(N - 1): B[i, i] = n
    for i in range(N - 1): B[N - 1, i] = t_val[i]
    B[N - 1, N - 1] = 1

    for i in range(N - 1): B[N, i] = u_val[i]
    B[N, N] = K
    
    LLL.reduction(B)

    # search the vector that has ±K as last element
    for row_idx in range(dim):
        sign = 1 if B[row_idx, N] == K else -1
        if abs(B[row_idx, N]) != K: continue
        
        alphas = []
        for j in range(N - 1):
            alphas.append( (-sign * B[row_idx, j]) % n )
        alphas.append( (sign * B[row_idx, N - 1]) % n ) 
        break
    
    # recover ki
    k = []
    for alpha, beta in zip(alphas, leak):
        k.append( alpha * pow(2, ell) + beta )
    
    # recover private key
    a = ((k[0] * s[0] - m[0]) * pow(r[0], -1, n)) % n 
    return k, a

def main():
    parser = argparse.ArgumentParser(description="ECDSA LLL attack on multi leakage")
    parser.add_argument(
        "--ip", type=str, default="127.0.0.1", metavar="h",
        help="Set the IP of the victim hosts (default=127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", type=int, default=4242, metavar="p",
        help="Set the PORT of the victim hosts (default=4242)"
    )

    parser.add_argument(
        "--number-of-sign", type=int, default=None, metavar="N",
        help="Set the number of message to read before the attack"
    )

    parser.add_argument(
        "--number-of-leak-bits", type=int, default=None, metavar="l",
        help="Set the number of leak bits of the vulnerable service"
    )

    args = parser.parse_args()
    cl = pwn.connect(args.ip, args.port)

    m, r, s, leak = [], [], [], []
    G, n, PK = get_G(cl), get_n(cl), get_pk(cl)
    
    for _ in range(args.number_of_sign):
        m_ = gen_msg(4)
        r_, s_, leak_ = sign_msg(cl, m_)
        m.append( int.from_bytes( hashlib.sha256(m_.encode()).digest(), "big" ) ); 
        r.append(r_); s.append(s_); leak.append(leak_)
    
    k, sk = perform_attack(m, r, s, leak, n, args.number_of_leak_bits)
    cl.sendlineafter(b"private key: ", str(sk).encode())
    tmp = cl.recvline().decode()
    print(f"{tmp}")

    cl.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
    except EOFError:
        print("secret key not found")
        exit(2)
    except Exception as e:
        print(f"error: {e}")
        exit(1)
