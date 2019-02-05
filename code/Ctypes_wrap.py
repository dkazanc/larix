#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
using Ctypes to access C flatsreg function
"""
import ctypes
from numpy.ctypeslib import ndpointer

def flatsregC():
    dll = ctypes.CDLL('c_function/flatsreg_core.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.flatsreg_main
    func.restype = None

    func.argtypes = [ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),  # input flat
                ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),  # input dark
                ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),  # input projection
                ctypes.c_int, # x
                ctypes.c_int, # y
                ctypes.c_int, # x1
                ctypes.c_int, # y1
                ctypes.c_int, # drift window size
                ctypes.c_int, # dimX
                ctypes.c_int, # dimY
                ctypes.c_int, # flats number
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # vector of errors
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # not corrected proj
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")] # corrected projection
    return func
