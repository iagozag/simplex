import argparse
import numpy as np

def makeMatrixFullRank(A):
    ''' Esta função recebe uma matriz, que pode ser em numpy,
        e retorna dois argumentos:
          - A matriz com linhas eliminadas
          - Uma lista indicando quais linhas foram eliminadas.
    '''
    if np.linalg.matrix_rank(A) == A.shape[0]: return A, []
    row = 1
    rowsEliminated = []
    counter = 0
    while 1:
        counter += 1
        B = A[0:(row+1), :]
        C = np.linalg.qr(B.T)[1]
        C[np.isclose(C, 0)] = 0
        if not np.any(C[row, :]):
            rowsEliminated.append(counter)
            A = np.delete(A, (row), axis=0)
        else:
            row += 1
        # end if
        if row >= A.shape[0]: break
    # end for
    return A, rowsEliminated
# end makeMatrixFullRank

def parse():
	parser = argparse.ArgumentParser(description="idk", epilog="idk2")

	parser.add_argument("filename")
	parser.add_argument("--decimals")
	parser.add_argument("--digits")
	parser.add_argument("--policy")
	
	return parser.parse_args()

def get_input():
	m = int(input())
	n = int(input())

	signs = []
	for i in range(m):
		signs.extend(map(int, input().split()))

	restrictions = []
	for i in range(n):
		line = input().split()
		row = list(map(int, line[:-2]))
		sign = -1 if line[-2] == "<=" else (0 if line[-2] == "=" else 1)
		last = int(line[-1])
		row.append(sign)
		row.append(last)

		restrictions.append(row)
		print(row)

	return m, n, signs, restrictions

def main():
	args = parse()
	m, n, signs, restrictions = get_input()


	

if __name__ == "__main__":
    main()
