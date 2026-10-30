import os, random, string

from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

class Param:
    def __init__(self, n: int, k: int, size: int = 32) -> None:
        '''
        @param: n, k must be in bits
        @param: size is the length in bytes
        '''
        if (n % k) != 0:
            raise TypeError(f"n % k = {n % k} not equal to 0")

        self.n = n
        self.k = k
        self.m = self.n // self.k 
        self.t = 2**self.m
        self.seed = os.urandom(size)

def G(param: Param, M: str | bytes) -> bytes:
    if isinstance(M, str): M = M.encode()

    digest = hashes.Hash(hashes.SHAKE256(digest_size=param.n // 8), backend=default_backend())
    digest.update(M)
    return digest.finalize()

def H(param: Param, M: str | bytes, C: bytes | None = None) -> tuple:
    if isinstance(M, str): M = M.encode()

    size = len(M) + len(C) if C is not None else len(M)
    n_size = param.n // 8
    if C is None:
        if len(M) >= n_size: raise TypeError("1. err in function H")
    else: 
        if size != n_size: raise TypeError("2. err in function H")

    C = os.urandom(n_size - len(M)) if C is None else C
    digest = hashes.Hash(hashes.SHA3_256(), backend=default_backend())
    digest.update(M + C)
    return digest.finalize(), C

def key_computation(param: Param, index: int) -> bytes:
    if not (0 <= index < param.t):
        raise ValueError(f"index {index} out of range [0, {param.t})")

    return HKDFExpand(
        algorithm=hashes.SHA256(),
        length=32,
        info=index.to_bytes(4, "big"),
    ).derive(param.seed)

def key_gen(param: Param) -> list:
    pk = []
    for i in range(param.t):
        pk.append( G(param, key_computation(param, i)) )
    return pk

def sign_message(param: Param, M: str) -> tuple:
    h, C = H(param, M)
    h_bin = bin(int.from_bytes(h, "big"))[2:].zfill(param.n)
    tmp = [ int("".join(h_bin[i:i+param.m]), 2) for i in range(0, len(h_bin), param.m) ]
    sign = []
    for el in tmp:
        sign.append(key_computation(param, el))

    return sign, C

def verify_message(param: Param, M: str | bytes, C: bytes, pk: list, sign: list) -> bool: 
    h, _ = H(param, M, C)
    h_bin = bin(int.from_bytes(h, "big"))[2:].zfill(param.n)
    tmp = [ int("".join(h_bin[i:i+param.m]), 2) for i in range(0, len(h_bin), param.m) ]
    for i, el in enumerate(tmp):
        if pk[el] != G(param, sign[i]):
            return False
    return True


def main():

    gen_msg = lambda x: "".join([ random.choice(string.ascii_letters) for _ in range(x) ])

    # toy version
    param = Param(16, 4)
    pk = key_gen(param)

    M = gen_msg(1)
    print(f"{M = }")

    sign, C = sign_message(param, M)
    assert verify_message(param, M, C, pk, sign)

    # real version
    # param = Param(256, 16)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"err: {e}")

