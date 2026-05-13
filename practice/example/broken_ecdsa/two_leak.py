import pwn
import sys
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
    return r, s

def perform_attack(
        m1: int, r1: int, s1: int,
        m2: int, r2: int, s2: int,
        n: int
    ) -> int:
    
    K, k1, k2 = 2 ** 128, None, None
    t = - (pow((s1 * r2), -1, n) * (s2 * r1)) % n
    u = pow((s1 * r2), -1, n) * (m2 * r1 - m1 * r2) % n
    B = IntegerMatrix(3, 3)
    B[0, 0] = n; B[0, 1] = 0; B[0, 2] = 0
    B[1, 0] = t; B[1, 1] = 1; B[1, 2] = 0
    B[2, 0] = u; B[2, 1] = 0; B[2, 2] = K

    LLL.reduction(B)
    for i in range(3):
        row = [B[i][j] for j in range(3)]
        print(f"b{i} = [{row[0]}, {row[1]}, {row[2]}]")
        if B[i][2] == K:
            k1, k2 = (- B[i][0]) % n, B[i][1]

    # restore a
    a = ((k1 * s1 - m1) * pow(r1, -1, n)) % n
    print(f"{a = }")
    return a


def main():
    IP = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    PORT = sys.argv[2] if len(sys.argv) > 2 else 4242
    cl = pwn.connect(IP, PORT)

    G, n, PK = get_G(cl), get_n(cl), get_pk(cl)
    r1, s1 = sign_msg(cl, "a")
    m1 = int.from_bytes( hashlib.sha256(b"a").digest(), "big" )
    r2, s2 = sign_msg(cl, "b")
    m2 = int.from_bytes( hashlib.sha256(b"b").digest(), "big" )
    
    sk = perform_attack(m1, r1, s1, m2, r2, s2, n)
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
