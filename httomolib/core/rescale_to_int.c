#include "rescale_to_int.h"

DLL int
rescale_float_to_int8(float* Input, unsigned char* Output, float input_min, float input_max, float factor, size_t total_elements)
{

    float     output_val;
    size_t    i;

    #pragma omp parallel for shared(Input, Output) private(i, output_val)
        for(i = 0; i < total_elements; i++)
        {
            output_val = Input[i];
            if (Input[i] < input_min) output_val = input_min;
            if (Input[i] > input_max) output_val = input_max;
            output_val -= input_min;
            output_val *= factor;
            Output[i] = (unsigned char) (output_val);
        }

    return 0;
}

DLL int
rescale_float_to_int16(float* Input, unsigned short* Output, float input_min, float input_max, float factor, size_t total_elements)
{

    float     output_val;
    size_t    i;

    #pragma omp parallel for shared(Input, Output) private(i, output_val)
        for(i = 0; i < total_elements; i++)
        {
            output_val = Input[i];
            if (Input[i] < input_min) output_val = input_min;
            if (Input[i] > input_max) output_val = input_max;
            output_val -= input_min;
            output_val *= factor;
            Output[i] = (unsigned short) (output_val);
        }

    return 0;
}

DLL int
rescale_float_to_int32(float* Input, unsigned int* Output, float input_min, float input_max, float factor, size_t total_elements)
{

    float     output_val;
    size_t    i;

    #pragma omp parallel for shared(Input, Output) private(i, output_val)
        for(i = 0; i < total_elements; i++)
        {
            output_val = Input[i];
            if (Input[i] < input_min) output_val = input_min;
            if (Input[i] > input_max) output_val = input_max;
            output_val -= input_min;
            output_val *= factor;
            Output[i] = (unsigned int) (output_val);
        }

    return 0;
}
