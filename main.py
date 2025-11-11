import argparse
import numpy as np


class Constants:
	def __init__(self):
		self.eps = 1e-12
		self.form_str = f"{{:>{Args.digits}.{Args.decimals}f}}"


class Args:
	def __init__(self):
		self.filename = ""
		self.decimals = 3
		self.digits = 1
		self.policy = 'largest'


Args = Args()
CONST = Constants()


def parse():
	parser = argparse.ArgumentParser(description="Simplex Solver", epilog="End of help")

	parser.add_argument('filename', type=str, help="Nome do arquivo lp de entrada")	
	parser.add_argument('--decimals', type=int, default=3, help="Número de casas decimais para imprimir valores numéricos.")
	parser.add_argument('--digits', type=int, default=1, help="Total de dígitos para imprimir valores numéricos.")
	parser.add_argument('--policy', choices=['largest', 'bland', 'smallest'], default='largest', help="Política a ser usada.")

	return parser.parse_args()


def printm(M):
    # Convert matrix to string with aligned columns
    formatted = [
        [CONST.form_str.format(value) for value in row]
        for row in M
    ]
    col_widths = [max(len(formatted[row][col]) for row in range(len(M))) for col in range(len(M[0]))]

    for row in formatted:
        for i, value in enumerate(row):
            print(value.rjust(col_widths[i]), end=" ")
        print()


def get_c(file, m, signs):
	c_line = file.readline().split()
	c_type = (-1 if c_line[0] == "min" else 1)

	c = []
	for i in range(m):
		c.append(int(c_line[i+1])*c_type)

		if signs[i] == -1:
			c[-1] *= -1
		elif signs[i] == 0:
			c.append(-int(c_line[i+1])*c_type)

	return c, c_type


def get_A(file, n, m, signs):
	A = []
	b = []
	equal = []
	for i in range(n):
		line = file.readline().split()
		sign = (1 if line[-2] != ">=" else -1)
		equal.append(1 if line[-2] == "==" else 0)

		A.append([]);
		for j in range(len(line)-2):
			A[i].append(int(line[j])*sign)
			if signs[j] == -1:
				A[i][-1] *= -1
			elif signs[j] == 0:
				A[i].append(-int(line[j])*sign)

		b.append(int(line[-1])*sign)

	m += signs.count(0)+1
	return A, b, equal, m


def to_fpi(n, m, A, c, b, equal):	
	for i in range(len(equal)):
		if not equal[i]:
			c.append(0)
			m += 1
			for j in range(n):
				A[j].append((1 if i == j else 0))

	for i in range(n):
		A[i].append(b[i])

	return n, m, A, c, b


def get_input():
	file = open(Args.filename, "r")

	m = int(file.readline().strip())
	n = int(file.readline().strip())

	signs = list(map(int, file.readline().split()))

	c, c_type = get_c(file, m, signs)

	A, b, equal, m = get_A(file, n, m, signs)

	n, m, A, c, b = to_fpi(n, m, A, c, b, equal)

	file.close()

	print("A:")
	printm(A)
	print()
	print("b:")
	print(np.array(b))
	print()
	print("c:")
	print(np.array(c))
	print()

	return n, m, c, A, signs, c_type


def vero(n, m, c, A):
	T = np.array([[0]*(m+n)]*(n+1))	
	T = T.astype(float)

	for i in range(n):
		T[i+1][i] = 1

	for i in range(m-1):
		T[0][i+n] = -c[i]

	for i in range(n):
		for j in range(m):
			T[i+1][j+n] = A[i][j]

	m += n
	n += 1

	print("Tableau VERO:")
	printm(T)
	print()

	return T, n, m


def get_base(T, n, m):
	X = set()
	Y = []
	for i in range(1, n):
		X.add(i)

	for i in range(n-1, m-1):
		cnt = 0
		idx = -1
		for j in range(1, n):
			if T[j][i] < -CONST.eps or T[j][i] > 1:
				cnt = 0
				break


			if T[j][i] == 1:
				cnt += 1
				idx = j
		if cnt == 1:
			if idx in X:
				X.remove(idx)
				Y.append(i)	
				T[0] -= T[0][i]*T[idx]

	return T, X, Y


def get_idx(T, n, m):
	mn = 0
	y = -1
	for i in range(n-1, m-1):
		if T[0][i] < -CONST.eps:
			if Args.policy == "bland":
				return i
			elif (Args.policy == "smallest" and (mn == 0 or mn < T[0][i])) or (Args.policy == "largest" and (mn == 0 or mn > T[0][i])):
					mn = T[0][i]
					y = i

	return y


def pivot(T, n, m, x, y):
	T[x] /= T[x][y]

	for i in range(0, n):
		if i != x and abs(T[i][y]) > CONST.eps:
			T[i] -= T[x]*T[i][y]


def simplex(T, n, m):
	# dual
	print("metodo dual:")
	while True:
		x = -1
		y = -1
		for i in range(1, n):
			if T[i][-1] < -CONST.eps:
				x = i

		if x < 0:
			print("terminou")
			print()
			break

		for i in range(n-1, m-1):
			if T[x][i] < -CONST.eps:
				y = i
				break

		if y < 0:
			print()
			return T, "inviavel"

		print("pivoteia", x, y)
		print()

		pivot(T, n, m, x, y);

		printm(T)
		print()

	# primal
	print("metodo primal:")
	while True:
		x = -1
		y = get_idx(T, n, m)

		if y < 0:
			print("terminou")
			print()
			break

		mn = 0
		for i in range(1, n): 
			if T[i][y] > CONST.eps and (mn == 0 or T[i][-1] / T[i][y] < mn):
				mn = T[i][-1] / T[i][y]
				x = i

		if x < 0:
			print()
			return T, "ilimitado"

		print("pivoteia", x, y)
		print()

		pivot(T, n, m, x, y);

		printm(T)
		print()

	return T, "otimo"


def get_sol(T, n, m, signs):
	T, X, Y = get_base(T, n, m)
	
	sol = {} 
	for y in Y:
		idx = -1
		for i in range(1, n):
			if T[i][y]:
				idx = i
				break
		sol[y] = T[idx][-1]

	sols = [0]*len(signs)
	j = n-1
	for i in range(len(signs)):
		if signs[i] == 0:
			sols[i] = (sol[j] if j in sol else 0)-(sol[j+1] if j+1 in sol else 0)
			j += 1
		else:
			sols[i] = (sol[j] if j in sol else 0)

		j += 1

	return sols


def get_dual(T, n, m):
	return T[0][:n-1]


def get_aux(T, n, m, X):
	b = []
	aux, n1, m1 = T, n, m
	aux = aux.tolist()
	
	for i in range(n):
		b.append(aux[i][-1])
		aux[i].pop()

	for i in range(n1-1, m-1):
		aux[0][i] = 0

	for x in X:
		m1 += 1
		aux[0].append(1)
		for i in range(1, n):
			if x != i:
				aux[i].append(0)
			else:
				aux[i].append(1)

	for i in range(n):
		aux[i].append(b[i])

	aux = np.array(aux)

	for x in X:
		aux[0] -= aux[x]

	return aux, n1, m1


def auxiliar(T, n, m):
	print("Auxiliar:")
	T, X, Y = get_base(T, n, m)

	if len(X) == 0:
		print("ja existe uma base para a matriz")
		print()
		return T, "otimo"

	aux, n1, m1 = get_aux(T, n, m, X)

	printm(aux)
	print()

	print("executando simplex:")
	aux, status = simplex(aux, n1, m1)

	if aux[0][-1] < 0:
		return T, "inviavel"

	return T, "otimo"


def calc(T, n, m):
	T, status = auxiliar(T, n, m)

	if status == "inviavel":
		return T, status

	T, status = simplex(T, n, m)
	
	return T, status


def main():
	args = parse()

	Args.filename = args.filename
	Args.decimals = args.decimals
	Args.digits = args.digits
	Args.policy = args.policy

	n, m, c, A, signs, c_type = get_input()

	T, n, m = vero(n, m, c, A)

	T, status = calc(T, n, m)

	print("Status: " + status)

	if status == "otimo": 
		print()
		print("tableau final:")
		printm(T)
		print()

		print("objetivo: ", end = "")
		print(CONST.form_str.format(T[0][-1]*c_type))

		print("solucao: ")
		sols = get_sol(T, n, m, signs)
		for x in sols:
			print(CONST.form_str.format(x), end = " ")
		print()


		print("dual: ")
		dual = get_dual(T, n, m)
		for x in dual:
			print(CONST.form_str.format(x), end = " ")
		print()


if __name__ == "__main__":
    main()
