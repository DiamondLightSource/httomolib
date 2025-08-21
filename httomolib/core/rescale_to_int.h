#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

#pragma once

#ifdef WIN32
#    define DLL __declspec(dllexport)
#else
#    define DLL
#endif

DLL int
rescale_float_to_int8(float* Input, unsigned char* Output, float input_min, float input_max, float factor, size_t total_elements);
DLL int
rescale_float_to_int16(float* Input, unsigned short* Output, float input_min, float input_max, float factor, size_t total_elements);
DLL int
rescale_float_to_int32(float* Input, unsigned int* Output, float input_min, float input_max, float factor, size_t total_elements);
