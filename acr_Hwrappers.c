#include "acr_Hwrappers.h"

void
inv_supermatrix(hlib_matrix_t A, hlib_acc_t acc){
	hlib_matrix_inv( A, acc, NULL );
}

void
gemm_supermatrix(double alpha, hlib_matrix_t A, hlib_matrix_t B, double beta, hlib_matrix_t C, hlib_acc_t acc) {
	// C := (alpha)AB+(beta)C
	hlib_matrix_mul (alpha,
					HLIB_MATOP_NORM, A,
					HLIB_MATOP_NORM, B,
					beta, C,
					acc, NULL);
}

void
gemv_supermatrix(double alpha, hlib_matrix_t A, pVec vec, double beta, pVec sol){
	// y := alpha*op(A) x + beta*y
	hlib_matrix_mulvec (alpha, A, vec->e, 
						beta, sol->e, 
						HLIB_MATOP_NORM, NULL);
}

void
geam_supermatrix(double alpha, hlib_matrix_t A, double beta, hlib_matrix_t B, hlib_acc_t acc) {
	// B = (alpha)A+(beta)B
	hlib_matrix_add (alpha, A,
					beta, B,
					acc, NULL);
}