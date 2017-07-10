#ifndef _acr_Hwrappers_h
#define _acr_Hwrappers_h

#include "acr_blockDiagonal.h"
#include <stdio.h>

void
inv_supermatrix(hlib_matrix_t A, hlib_acc_t acc);

void
gemm_supermatrix(double alpha, hlib_matrix_t A, hlib_matrix_t B, double beta, hlib_matrix_t C, hlib_acc_t acc);

void
gemv_supermatrix(double alpha, hlib_matrix_t A, pVec vec, double beta, pVec sol);

void
geam_supermatrix(double alpha, hlib_matrix_t A, double beta, hlib_matrix_t B, hlib_acc_t acc);

#endif