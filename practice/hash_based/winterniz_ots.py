import hashlib
import os
import math
import secrets


class WOTSParam:
    
    def __init__(self, n: int, w: int):
        self.s = 256
        self.n, self.w = n * 8, w
        self.l1 = self.n // w
        self.l2 = math.floor( math.log2(self.l1 * (2**w - 1)) / w ) + 1
        self.l = self.l1 + self.l2

def G(M: str | bytes) -> bytes:
    name = "sha3_256"
    if isinstance(M, str):
        M = M.encode()
    
    if len(M) != 32:
        raise TypeError("G: {0, 1}^{256} -> {0, 1}^{256}")

    g = hashlib.new(name)
    g.update(M)
    return g.digest()

def H(param: WOTSParam, M: str | bytes) -> bytes:
    name = "shake_256"
    if isinstance(M, str):
        M = M.encode()

    g = hashlib.new(name)
    g.update(M)
    return g.digest(param.n // 8) # type: ignore

def key_generation(param: WOTSParam) -> tuple:
    
    sk, pk, hash_matrix = [], [], []
    for i in range(0, param.l):
        size = param.s // 8 
        x_i = int.to_bytes( secrets.randbits(param.s), length=size, byteorder="big" )
        hash_chain_i = [ x_i ]
        for j in range(1, 2**param.w - 1):
            hash_chain_i.append( G(hash_chain_i[-1]) )
        hash_matrix.append((hash_chain_i[1:]))
        sk.append(x_i); pk.append(G(hash_chain_i[-1]))

    return sk, pk, hash_matrix

def sign_msg(param: WOTSParam, sk: list, hash_matrix: list, pk: list, M: str | bytes) -> list:

    if isinstance(M, str):
        M = M.encode()
    
    h = H(param, M)
    chunks = [ bin(int.from_bytes(h, byteorder="big"))[2:].rjust(param.n, '0')[i:i+param.w] for i in range(0, param.n, param.w) ]
    
    sign, csum = [], 0 
    for jump, priv, pub, hc in zip(chunks, sk, pk, hash_matrix):
        hash_chain = [priv, *hc, pub]
        sign.append(hash_chain[ int(jump, 2) ])
        csum += (2 ** param.w - 1 - int(jump, 2))

    tmp = [ bin(csum)[2:].rjust(param.w * param.l2, '0')[i:i+param.w] for i in range(0, param.w * param.l2, param.w) ]
    for i in range(param.l1, param.l):
        hash_chain = [ sk[i], *hash_matrix[i], pk[i] ]
        sign.append(hash_chain[ int(tmp[i - param.l1], 2) ])

    return sign

def verify_msg(param: WOTSParam, sign: list, pk: list, M: str | bytes) -> bool:

    if isinstance(M, str):
        M = M.encode()

    h = H(param, M)
    chunks = [ bin(int.from_bytes(h, byteorder="big"))[2:].rjust(param.n, '0')[i:i+param.w] for i in range(0, param.n, param.w) ]
    csum = sum([ 2 ** param.w - 1 - int(el, 2) for el in chunks ])
    
    # check of the csum
    tmp = [ bin(csum)[2:].rjust(param.w * param.l2, '0')[i:i+param.w] for i in range(0, param.w * param.l2, param.w) ]
    for i in range(param.l1, param.l):
        sign_i = sign[i]
        for j in range(2 ** param.w - int(tmp[i - param.l1], 2) - 1):
            sign_i = G(sign_i)
        if sign_i != pk[i]: return False

    # check of sign
    for i, jump in enumerate(chunks):
        x = 2 ** param.w - 1 - int(jump, 2)
        sign_i = sign[i]
        for j in range(x):
            sign_i = G(sign_i)

        if sign_i != pk[i]: return False

    return True

def main():
    import random, string
    gen_word = lambda x: "".join([ random.choice(string.ascii_letters) for _ in range(x) ])
    
    # toy version
    param = WOTSParam(n=2, w=2)
    sk, pk, hash_matrix = key_generation(param)
    M = gen_word(7)
    sign = sign_msg(param, sk, hash_matrix, pk, M)
    assert verify_msg(param, sign, pk, M)


    # regular version
    param = WOTSParam(n=256, w=8)

    word = gen_word(256)
    H(param, word)

if __name__ == "__main__":
    main()

