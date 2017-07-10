#ifndef _acr_mpi_h
#define _acr_mpi_h

#include <hlib.hh>
#include "hlib-c.h"
#include <mpi.h>

using namespace HLIB;

// MPI Type Struct to collect data info
typedef struct dataSplitInfo2m {
	int middleIdx;
	int sizeFH; 
	int sizeSH; 
} dsi2m;

// MPI Type Struct to collect data info
typedef struct dataSplitInfo4m {
	int sizem1; 
	int sizem2; 
	int sizem3; 
	int sizem4; 
} dsi4m;

void
MPI_print(MPI_Comm comm, char const *MESSAGE, int num, int verbose);

void
MPI_blank_line(MPI_Comm comm);

void
probe_HMatrix(MPI_Comm comm, hlib_matrix_t A, int MPI_rank, char const *NAME, int level, int block);

void
probe_HVector(MPI_Comm comm, hlib_vector_t vec, int MPI_rank, char const *NAME, int level, int block);

void 
MPI_Send_HMatrix_1m(MPI_Comm comm, TMatrix* A, int destination);

TMatrix*
MPI_Recv_Matrix_1m(MPI_Comm comm, int source);

void
MPI_Send_HMatrix_2m(MPI_Comm comm, TMatrix* A, int destination);

TMatrix*
MPI_Recv_Matrix_2m(MPI_Comm comm, int source);

void
MPI_Send_HMatrix(MPI_Comm comm, TMatrix* A, int destination);

TMatrix*
MPI_Recv_Matrix(MPI_Comm comm, int source);

void
MPI_Send_HVector(MPI_Comm comm, TVector* vec, int destination);

TVector*
MPI_Recv_Vector(MPI_Comm comm, int source,int blockSize);

#endif