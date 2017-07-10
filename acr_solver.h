#ifndef _acr_solver_h
#define _acr_solver_h

#include "acr_blockDiagonal.h"
#include "hlib-c.h"
#include <mpi.h>

int ranksToScatter(int q, int level, int i);

int oddsAtLevel(int q, int level, int i, int last);

int oddsToInvert(int q, int level, int i, int last);

void solve_ACR_parallel(MPI_Comm comm, pcrDiag E, pcrDiag D, pcrDiag F, pcrDiag AUXD1, pcrDiag AUXD2,
hlib_matrix_t blankMat, pcrVec bK, pcrVec levSol, pcrVecLevel gatherSol,
int q, const int crDim, int block_nx, hlib_acc_t acc, double *ELIM_MEM, int flag_ACR_iters, int dispChecks, int dispTime, hlib_permutation_t perm);

void acr_setup(MPI_Comm comm, pcrDiag E, pcrDiag D, pcrDiag F, pcrDiag AUXD1, pcrDiag AUXD2,
hlib_matrix_t blankMat, pcrVec bK, int q, const int crDim, int block_nx, hlib_acc_t acc, double *ELIM_MEM, int flag_ACR_iters, int dispMemory, int dispTime);

double acr_apply(MPI_Comm comm,pcrDiag E, pcrDiag D, pcrDiag F, pcrVec bK, int q, int crDim, int block_nx, pcrVec levSol, pcrVecLevel gatherSol, int flag_ACR_iters, int dispTime, hlib_permutation_t perm);

#endif