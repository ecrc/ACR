#include <hlib.hh>
#include "acr_blockDiagonal.h"
#include "hlib-c.h"
#include "mpi.h"
#include <assert.h>
#include "acr_Hwrappers.h"

hlib_matrix_t
build_supermatrix(MPI_Comm comm, int cr_n, int block_nx, int con, hlib_acc_t acc, hlib_blockclustertree_t bct, int type, 
	char *folder){
    int             info;
    hlib_matrix_t   A = NULL, S = NULL;
    char 			buffer_D[120];
    char 			buffer_E[120];
    char 			buffer_F[120];
	int world_rank;
	MPI_Comm_rank(comm, &world_rank);

    if (type == 3) {
		sprintf(buffer_D, "%sD%d.mat",folder,con);
		S = hlib_load_matrix( buffer_D, &info );
		if(info == 0){
			((HLIB::TMatrix *) S)->set_unsymmetric();
			A = hlib_matrix_build_sparse( bct, S, acc, &info );
			hlib_matrix_free( S, &info );
		}
		else{ printf("Error at building D[lev][%d] info = %d\n",con,info); }
	}
	else if (type == 1) {
		sprintf(buffer_E, "%sE%d.mat",folder,con);
		S = hlib_load_matrix( buffer_E, &info );
		if(info == 0){
			((HLIB::TMatrix *) S)->set_unsymmetric();
			A = hlib_matrix_build_sparse( bct, S, acc, &info );
			hlib_matrix_free( S, &info );
		}
		else if (con!=0) { printf("Error at building E[lev][%d] info = %d\n",con,info); }
	}
	else if (type == 2) {
		sprintf(buffer_F, "%sF%d.mat",folder,con);
		S = hlib_load_matrix( buffer_F, &info );
		if(info == 0){
			((HLIB::TMatrix *) S)->set_unsymmetric();
			A = hlib_matrix_build_sparse( bct, S, acc, &info );
			hlib_matrix_free( S, &info );
		}
		else if (con != cr_n-1) { printf("Error at building F[lev][%d] info = %d\n",con,info); }
	}
	else {
		printf("GC: Wrong matrix type <1,2,3>, Specified type = %d\n", type);
		return NULL;
	}
    return A;
}

hlib_matrix_t 
load_supermatrix(int con, hlib_acc_t acc, hlib_blockclustertree_t bct, int type, char *folder){
	
    int                         info;
    hlib_matrix_t               A = NULL, S = NULL;
    char 						buffer_D[60];
    char 						buffer_E[60];
    char 						buffer_F[60];
    
    if (type == 3) {
		sprintf(buffer_D, "%sD%d.mat",folder,con);
		S = hlib_load_matrix( buffer_D, &info );
		A   = hlib_matrix_build_sparse( bct, S, acc, &info );
	    hlib_matrix_free( S, &info );
	}
	else if (type == 1) {
		sprintf(buffer_E, "%sE%d.mat",folder,con);
	    S = hlib_load_matrix( buffer_E	, &info );
	    A   = hlib_matrix_build_sparse( bct, S, acc, &info );
	    hlib_matrix_free( S, &info );
	}
	else if (type == 2) {
		sprintf(buffer_F, "%sF%d.mat",folder,con);
	    S = hlib_load_matrix( buffer_F	, &info );
	    A   = hlib_matrix_build_sparse( bct, S, acc, &info );
	    hlib_matrix_free( S, &info );
	}
	else {
		printf("This option is not supported\n");
		printf("type = %d\n",type);
		return NULL;
	}
    return A;
}

hlib_matrix_t
set_blankMat(char *fblank, int nx, double h, double eta, double nmin, hlib_blockclustertree_t *bct, hlib_acc_t acc){

	hlib_matrix_t blankMat = NULL;
	hlib_coord_t  coord;
	double **  vertices;
	hlib_clustertree_t          ct;
    hlib_admcond_t              adm;
	vertices = (double**) malloc( (nx*nx) * sizeof(double*) );
	assert(vertices!=NULL);
	int con=0, iloop, jloop;
	for ( iloop = 0; iloop < nx; iloop++ ) {
	    for ( jloop = 0; jloop < nx; jloop++ ) {
	        vertices[con]    = (double*) malloc( sizeof(double)*2 );
	        vertices[con][0] = h * ((double) jloop);
	        vertices[con][1] = h * ((double) iloop);
	        con++;
	    }
	}
	coord = hlib_coord_import( nx*nx, 2, vertices, NULL, NULL );
	ct = hlib_clt_build_bsp( coord, HLIB_BSP_GEOM_REG, nmin, NULL );
	adm = hlib_admcond_geom( HLIB_ADM_AUTO, eta, NULL );
	*bct = hlib_bct_build( ct, ct, adm, NULL );
    hlib_matrix_t spablankMat = NULL;
	spablankMat = hlib_load_matrix( fblank, NULL );
	((HLIB::TMatrix *) spablankMat)->set_unsymmetric();
	blankMat = hlib_matrix_build_sparse( *bct, spablankMat, acc, NULL );
	gemm_supermatrix(0.0, blankMat, blankMat, 0.0, blankMat, acc);
	return blankMat;
}

void 
set_A0(MPI_Comm comm, int cr_n, pcrDiag E, pcrDiag D, pcrDiag F, hlib_acc_t acc, hlib_matrix_t blankMat,
hlib_blockclustertree_t bct, char *folder, int dispMem, double *SETUP_MEM){
	int 	type;
	int 	block_nx = D->level[0]->block[0]->size;
	int 	world_rank;
	MPI_Comm_rank(comm, &world_rank);
	double LOCAL_MEM_D = 0, LOCAL_MEM_E = 0, LOCAL_MEM_F = 0;
	double GLOBAL_MEM_D = 0,GLOBAL_MEM_E = 0,GLOBAL_MEM_F = 0;

	type = 3;
	D->level[0]->block[world_rank]->e = build_supermatrix(comm, cr_n, block_nx, world_rank, acc, bct, type, folder);
	LOCAL_MEM_D = get_Matrix_Size_MegaBytes(D->level[0]->block[world_rank]->e);

	type = 1;
	if(world_rank==0) {
		E->level[0]->block[world_rank]->e = hlib_matrix_copy(blankMat, NULL);
		LOCAL_MEM_E = get_Matrix_Size_MegaBytes(E->level[0]->block[world_rank]->e);
	}
	else{
		E->level[0]->block[world_rank]->e = build_supermatrix(comm, cr_n, block_nx, world_rank, acc, bct, type, folder);
		LOCAL_MEM_E = get_Matrix_Size_MegaBytes(E->level[0]->block[world_rank]->e);
	}

	type = 2;
	if(world_rank==cr_n-1) { // Managing exception, F_last does not exist, put a zero matrix
		F->level[0]->block[cr_n-1]->e = hlib_matrix_copy(blankMat, NULL);
		LOCAL_MEM_F = get_Matrix_Size_MegaBytes(F->level[0]->block[world_rank]->e);
	}
	else{
		F->level[0]->block[world_rank]->e = build_supermatrix(comm, cr_n, block_nx, world_rank, acc, bct, type, folder);
		LOCAL_MEM_F = get_Matrix_Size_MegaBytes(F->level[0]->block[world_rank]->e);
	}

	MPI_Reduce(&LOCAL_MEM_D, &GLOBAL_MEM_D, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
	MPI_Reduce(&LOCAL_MEM_E, &GLOBAL_MEM_E, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
	MPI_Reduce(&LOCAL_MEM_F, &GLOBAL_MEM_F, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	if(dispMem==1 && world_rank==0){
		*SETUP_MEM = GLOBAL_MEM_D+GLOBAL_MEM_E+GLOBAL_MEM_F;
		printf("   MEM_Level_0(E+D+F)_MB = %1.2f\n\n",*SETUP_MEM);
	}
}

void 
set_RHS(MPI_Comm comm, char *folder, hlib_permutation_t perm, pcrVec rhs){
	int world_rank;
	MPI_Comm_rank(comm, &world_rank);
	char file_rhs_i[250];
	sprintf(file_rhs_i, "%srhs_%d.mat", folder, world_rank);
	rhs->level[0]->block[world_rank]->e = hlib_load_vector( file_rhs_i, NULL ); 
	hlib_vector_permute(rhs->level[0]->block[world_rank]->e, perm, NULL);
}