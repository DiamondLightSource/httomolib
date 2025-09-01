cdef extern from "rescale_to_int.h":
    int rescale_float_to_int8(float* Input, unsigned char* Output, float input_min, float input_max, float factor, size_t total_elements)
    int rescale_float_to_int16(float* Input, unsigned short* Output, float input_min, float input_max, float factor, size_t total_elements)
    int rescale_float_to_int32(float* Input, unsigned int* Output, float input_min, float input_max, float factor, size_t total_elements)
