#!/usr/bin/env python
import hashlib
import json
import secrets
from Crypto.PublicKey import ECC

import sys
import os
import argparse 


FLAG = os.getenv("FLAG", "flag{REDACTED}")


def key_generation() -> tuple:
    CURVE = "P-256"
    param = ECC._curves[CURVE]
    n = int(param.order)
    G = ECC.EccPoint(param.Gx, param.Gy, curve=CURVE)
    key = ECC.generate(curve=CURVE)
    return (G, n, int(key.d), key.pointQ)

def sign_msg(a: int, G: ECC.EccPoint, n: int, M: str, leak_bits: int = None) -> tuple:
    m = int.from_bytes( hashlib.sha256(M.encode()).digest(), "big" )
    r, s, beta = 0, 0, None

    while r == 0 or s == 0:
        k = secrets.randbelow(2**125 - 1) + 1 if leak_bits is None else secrets.randbelow(n - 1) + 1
        R = k * G
        r = int(R.x) % n
        if r == 0:
            continue
        k_inv = pow(k, -1, n)
        s = (k_inv * (m + a * r)) % n
        beta = k % (2**leak_bits) if leak_bits is not None else None

    return (r, s, beta)

def verify(PK: ECC.EccPoint, G: ECC.EccPoint, n: int, M: str, r: int, s: int) -> bool:
    m = int.from_bytes( hashlib.sha256(M.encode()).digest(), "big" )
    if not (1 <= r < n and 1 <= s < n):
        return False

    s_inv = pow(s, -1, n)
    u1 = (m * s_inv) % n
    u2 = (r * s_inv) % n

    P = u1 * G + u2 * PK
    return int(P.x) % n == r

def two_leak_main():
    G, n, sk, pk = key_generation()
    print(f"G = ({hex(G.x)}, {hex(G.y)})")
    print(f"{n = }")
    print(f"{sk = }", file=sys.stderr)
    print(f"PK = ({hex(pk.x)}, {hex(pk.y)})")
    for _ in range(2):
        msg = input("choose the message you want to sign: ")
        r, s, _ = sign_msg(sk, G, n, msg)
        print(f"sign [r, s]: ({hex(r)}, {hex(s)})")

    a = int(input("private key: "))
    if a == sk:
        print(f"You won! {FLAG}")

def multi_leak_main(N: int, leak_bits: int):
    G, n, sk, pk = key_generation()
    print(f"G = ({hex(G.x)}, {hex(G.y)})")
    print(f"{n = }")
    print(f"{sk = }", file=sys.stderr)
    print(f"PK = ({hex(pk.x)}, {hex(pk.y)})")
    for _ in range(N):
        msg = input("choose the message you want to sign: ")
        r, s, beta = sign_msg(sk, G, n, msg, leak_bits)
        print(f"sign [r, s, beta]: ({hex(r)}, {hex(s)}, {hex(beta)})")

    a = int(input("private key: "))
    if a == sk:
        print(f"You won! {FLAG}")


def main():
    parser = argparse.ArgumentParser(description="ECDSA LLL Challenge")
    parser.add_argument(
        "--multi-leak", type=int, default=None, metavar="N",
        help="Number of signatures for the multi-leak attack"
    )
    parser.add_argument(
        "--leak-bits", type=int, default=8, metavar="L",
        help="Number of leaked LSBs per nonce (default = 8)"
    )
    args = parser.parse_args()

    if args.multi_leak is not None:
        multi_leak_main(args.multi_leak, args.leak_bits)
    else:
        two_leak_main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("exit")
    except Exception as e:
        print(f"err: {e}")
