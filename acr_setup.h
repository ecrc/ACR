#ifndef _acr_setup_h
#define _acr_setup_h

#include "acr_blockDiagonal.h"
#include "hlib-c.h"

hlib_matrix_t
build_supermatrix(MPI_Comm comm, int cr_n, int block_nx, int con, hlib_acc_t acc, hlib_blockclustertree_t bct, int type, 
	char *folder);

hlib_matrix_t
load_supermatrix(int con, hlib_acc_t acc, hlib_blockclustertree_t bct, int type, char *folder);

hlib_matrix_t
set_blankMat(char *fblank, int nx, double h, double eta, double nmin, hlib_blockclustertree_t *bct, hlib_acc_t acc);

void 
set_A0(MPI_Comm comm, int crDim, pcrDiag E, pcrDiag D, pcrDiag F, hlib_acc_t acc, hlib_matrix_t blankMat,
hlib_blockclustertree_t bct, char *folder, int dispMem, double *SETUP_MEM);

void 
set_RHS(MPI_Comm comm, char *folder, hlib_permutation_t perm, pcrVec rhs);

#endif