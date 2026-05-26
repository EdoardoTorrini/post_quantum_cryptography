import hashlib
import os


H, G = "sha3_256", "sha3_256"
KDF = "shake_256"
n = 256

def compute_hexdigest(name: str, M: str | bytes, n: int = 32) -> str:
    if isinstance(M, str):
        M = M.encode()

    g = hashlib.new(name)
    g.update(M)
    try:
        return g.hexdigest()
    except TypeError:
        return g.hexdigest(n) # type: ignore

def key_generation(seed: int) -> tuple:
    sk, pk = [], []
    for j in range(n):
        x0j =  compute_hexdigest(KDF, f"{seed}-0-{j}")
        x1j =  compute_hexdigest(KDF, f"{seed}-1-{j}")

        sk.append([x0j, x1j])
        pk.append([
            compute_hexdigest(G, x0j),
            compute_hexdigest(G, x1j)
        ])
    return sk, pk


def sign_msg(M: str, sk: list) -> list:
    sign = []
    for j, bit in enumerate( bin(int(compute_hexdigest(H, M), 16))[2:].rjust(n, '0') ):
         sign.append( sk[j][int(bit)] )

    return sign

def verify_msg(M: str, pk: list, sign: list) -> bool:
    for j, bit in enumerate( bin(int(compute_hexdigest(H, M), 16))[2:].rjust(n, '0') ):
        if compute_hexdigest(G, sign[j]) != pk[j][ int(bit) ]:
            return False
    return True

def main():
    seed, M = int.from_bytes( os.urandom(32), byteorder="big"), "prova"
    sk, pk = key_generation(seed)
    sign = sign_msg(M, sk)
    assert verify_msg(M, pk, sign)

if __name__ == "__main__":
    main()

