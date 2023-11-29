# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np

loaded_from_source = r'''

extern "C"{

//------------------------------------------------------------------------------------------------------------------------
// Device Functions
//------------------------------------------------------------------------------------------------------------------------

__global__ void device_add_tab(float* d_tabA, float* d_tabB, float* d_tabApB, int size) {

	int ijk = blockDim.x * blockIdx.x + threadIdx.x;
	if (ijk < size) {
		d_tabApB[ijk] = d_tabA[ijk] + d_tabB[ijk];
	}
}

}'''

module = cp.RawModule(code=loaded_from_source)
device_add_tab = module.get_function('device_add_tab')
size = 3000
n_threads = 1024
n_blocks = (size) // n_threads + 1

print(n_threads)
print(n_blocks)

#test fonction __global__
#allocation gpu
d_tab_A = cp.full(shape = size, fill_value= 1.0, dtype= cp.float32)
d_tab_B = cp.full(shape = size, fill_value= 2.0, dtype= cp.float32)
d_tab_C = cp.full(shape = size, fill_value= 0.0, dtype= cp.float32)


device_add_tab((n_blocks,), (n_threads,), (d_tab_A, d_tab_B, d_tab_C, cp.int32(size)))

result = cp.asnumpy(d_tab_C)
print(result)