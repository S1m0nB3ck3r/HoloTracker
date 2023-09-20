import cupy as cp
import numpy as np
from cupyx import jit

@jit.rawkernel()
def d_calc_phase(d_plan_complex, d_phase, size_x, size_y):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = size_x * size_y

    jj = index // size_x
    ii = index - jj * size_x

    if (ii < size_x and jj < size_y):
        cplx = d_plan_complex[0,0]
        r = cp.real(cplx)
        if (r == 0.0):
            d_phase[ii, jj] = 0.0
        elif(r > 0.0):
            d_phase[ii, jj] = cp.arctan(cp.imag(cplx) / cp.real(cplx))
        else:
            d_phase[ii, jj] = cp.pi + cp.arctan(cp.imag(cplx) / cp.real(cplx))


if __name__ == '__main__':
    # Définir les dimensions de la matrice
    size_x = 1024
    size_y = 1024

    # Initialiser la matrice d_plan_complex avec des nombres complexes aléatoires
    d_plan_complex = cp.random.rand(size_x, size_y) + 1j * cp.random.rand(size_x, size_y)

    # Initialiser la matrice d_phase avec des zéros
    d_phase = cp.zeros((size_x, size_y))

    # Définir la taille des blocs et des grilles pour la fonction CUDA
    block_size = 128
    grid_size = (size_x * size_y + block_size - 1) // block_size

    # Appeler la fonction CUDA pour calculer la phase
    d_calc_phase[grid_size, block_size](d_plan_complex, d_phase, size_x, size_y)

    # Afficher la matrice d_phase
    print(d_phase)

    