cdef extern from "rescale_to_int.h":
    int rescale_float_to_int8(float* Input, unsigned char* Output, float input_min, float input_max, float factor, size_t total_elements)
    int rescale_float_to_int16(float* Input, unsigned short* Output, float input_min, float input_max, float factor, size_t total_elements)
    int rescale_float_to_int32(float* Input, unsigned int* Output, float input_min, float input_max, float factor, size_t total_elements)

cdef extern from "data_check.h":
    int count_zeros_16bit_data(unsigned short* Input, float* Output, size_t total_elements)
    int count_zeros_32bit_float_data(float* Input, float* Output, size_t total_elements)
    int check_nans_infs_32bit_float_data(float* Input, unsigned char* ifnaninfs_present, size_t total_elements)