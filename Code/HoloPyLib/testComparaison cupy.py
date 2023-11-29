# -*- coding: utf-8 -*-

import numpy as np
import cupy as cp



arrX = cp.linspace(0.0, 100.0, num = 100, dtype = np.float32)

arrBool = arrX > 50.0
h_arrBool = cp.asnumpy(arrBool)

print(arrBool)



arrY = np.linspace(0.0, 10.0, num = 100, dtype = np.float32)

