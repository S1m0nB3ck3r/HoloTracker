# -*- coding: utf-8 -*-

import cupy as cp
from typeHolo import *
from cupyx import jit
import cc3d
import time



@jit.rawkernel()
def cuda_Binaries_Focus_Volume(d_bin_volume, d_focus_volume, threshold, sizeX, sizeY, sizeZ):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    kk = index // planSize
    jj = ( index - kk * planSize )// sizeX
    ii = index - jj * sizeX - kk * planSize

    if (ii < sizeX and jj < sizeY and kk < sizeZ):

        if d_focus_volume[ii, jj, kk] > threshold:
            d_bin_volume[ii, jj, kk] = True
        else:
            d_bin_volume[ii, jj, kk] = False

    jit.syncthreads()



#test
size = 100
focus_volume = cp.random.rand(size, size, size, dtype = cp.float32)
bin_volume = cp.full(shape=(size,size, size), dtype=cp.bool_, fill_value=False)
nthreads =  1024
nblocks = (size * size * size) // 1024 +1

cuda_Binaries_Focus_Volume[nblocks, nthreads](bin_volume, focus_volume, 0.5, size, size, size)

h_bin_volume = cp.asnumpy(bin_volume)
t = time.perf_counter()
labels_out, number_of_labels = cc3d.connected_components(h_bin_volume, connectivity=6, return_N=True) # 26-connected
print(time.perf_counter() - t)
print(number_of_labels)


