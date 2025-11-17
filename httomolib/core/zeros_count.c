#include "zeros_count.h"

DLL int
count_zeros_16bit_data(unsigned short* Input, float* Output, size_t total_elements)
{

    size_t    i;
    size_t    total_number_zeros;

    total_number_zeros = 0;

        for(i = 0; i < total_elements; i++)
        {
            if (Input[i] == 0) {total_number_zeros++;}
        }
        Output[0] = (float) (total_number_zeros);

    return 0;
}

DLL int
count_zeros_32bit_float_data(float* Input, float* Output, size_t total_elements)
{

    size_t    i;
    size_t    total_number_zeros;

    total_number_zeros = 0;

        for(i = 0; i < total_elements; i++)
        {
            if (Input[i] == 0.0f) {total_number_zeros++;}
        }
        Output[0] = (float) (total_number_zeros);

    return 0;
}
