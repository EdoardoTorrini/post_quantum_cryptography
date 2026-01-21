

def inf_norm(t):
    all_coeffs = [c.lift_centered().abs() for row in t for c in row[0].list()]
    return max(all_coeffs) if all_coeffs else 0


q, n = 541, 4
k, l = 3, 2

n_1, n_2 = 3, 2

# definisco il Galois Field
Zq = GF(q)

# Creo il polynomial ring su Zq
P.<x> = PolynomialRing(Zq)

# Definisco l'anello quoziente
R.<x> = P.quotient(x^n + 1)


A = matrix(R, [
    [442 + 502*x + 513*x^2 + 15*x^3,   368 + 166*x + 37*x^2 + 135*x^3],
    [479 + 532*x + 116*x^2 + 41*x^3,   12 + 139*x + 385*x^2 + 409*x^3],
    [29 + 394*x + 503*x^2 + 389*x^3,    9 + 499*x + 92*x^2 + 254*x^3]
])

s = matrix(R, [
    [2 - 2*x + x^3],
    [3 - 2*x - 2*x^2 - 2*x^3]
])

e = matrix(R, [
    [2 - 2*x - x^2],
    [1 + 2*x + 2*x^2 + x^3],
    [-2 - x^2 - 2*x^3]
])

t = A * s + e
print(f"t = A * s + e = \n{t}")
print(f"||t|| inf: {inf_norm(t)}")