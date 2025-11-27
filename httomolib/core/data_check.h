#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

#pragma once

#ifdef WIN32
#    define DLL __declspec(dllexport)
#else
#    define DLL
#endif

DLL int
count_zeros_16bit_data(unsigned short* Input, float* Output, size_t total_elements);
DLL int
count_zeros_32bit_float_data(float* Input, float* Output, size_t total_elements);
DLL int
check_nans_infs_32bit_float_data(float* Input, unsigned char* ifnaninfs_present, size_t total_elements);