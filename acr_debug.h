#ifndef _acr_debug_h
#define _acr_debug_h

#include "acr_blockDiagonal.h"
#include <stdio.h>

void helloWorld();

double get_Matrix_Size_MegaBytes(hlib_matrix_t A);

size_t get_Matrix_Memory(char const *NAME, hlib_matrix_t A);

size_t get_Vector_Memory(char const *NAME, hlib_vector_t vec);

void get_Bytes_Memory(char const *NAME, size_t bytes);


// Debug functions
void 
print_Hmatrix_dense_short(hlib_matrix_t A, char const *NAME);

void 
print_Hmatrix_dense(hlib_matrix_t A, char const *NAME);

void 
print_Vector(hlib_vector_t vec, char const *NAME);

void
visualize_HMatrix(hlib_matrix_t A, char const *NAME,int number);

void
visualize_H_Matrix(char const *NAME, hlib_matrix_t H_Mat, int vis_code);

void
checkSymmetry(hlib_matrix_t A, char const *NAME);

#endif
