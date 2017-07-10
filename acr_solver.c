#include "acr_solver.h"
#include "acr_Hwrappers.h"
#include <hlib.hh>
#include "acr_mpi.h"
#include <mpi.h>

int
ranksToScatter(int q, int level, int i){
	int divider =  (int) pow(2.0 , level);
	int skipSize = divider*2;
	if ( (i%skipSize == (divider-1) )  ){
		return i;
	}
	return -1;
}

int
oddsAtLevel(int q, int level, int i, int last){
	int divider = (int) pow(2.0 , level);

	if ( i%divider == (divider-1) ){
		if ( last==0 ){  
			return i;
		}
		else if(last == 1){
			if( i == (int) pow(2.0 , q) - 1 ){
				return -1;
			}
			else{
				return i;
			}
		}
		else if(last == -1){
			if( i == (int) pow(2.0 , level) - 1 ){
				return -1;
			}
			else{
				return i;
			}
		}
	}
	return -1;
}

int
oddsToInvert(int q, int level, int i, int last){
	int divider = (int) pow(2.0 , level);
	if ( i%divider == (divider-1)/2 ){
		if ( last==0 ){  
			return i;
		}
		else{
			if( i == ( (int)pow(2.0,level-1)-1) || i==0 ){
				return -1;
			}
			else{
				return i;
			}
		}
	}
	return -1;
}

void solve_ACR_parallel(MPI_Comm comm, pcrDiag E, pcrDiag D, pcrDiag F, pcrDiag AUXD1, pcrDiag AUXD2,
hlib_matrix_t blankMat, pcrVec bK, pcrVec levSol, pcrVecLevel gatherSol,
int q, const int crDim, int block_nx, hlib_acc_t acc, double *ELIM_MEM_MB, int flag_ACR_iters, int dispMemory, int dispTime, hlib_permutation_t perm){
	acr_setup(comm, E, D, F, AUXD1, AUXD2, blankMat, bK, q, crDim, block_nx, acc, ELIM_MEM_MB, flag_ACR_iters, dispMemory, dispTime);
	acr_apply(comm, E, D, F, bK, q, crDim, block_nx, levSol, gatherSol, flag_ACR_iters, dispTime, perm);
}

void acr_setup(MPI_Comm comm, pcrDiag E, pcrDiag D, pcrDiag F, pcrDiag AUXD1, pcrDiag AUXD2,
hlib_matrix_t blankMat, pcrVec bK, int q, const int crDim, int block_nx, hlib_acc_t acc, 
double *ELIM_MEM, int flag_ACR_iters, int dispMemory, int dispTime){

	int                         i,iiq;
	int                         cEv,cOd,cr_n;
	int 						iiqK;

	double 						timerCR_ALL=0.0, timerCR_ALL_MAX=0.0;
	double 						timerCR_LEV=0.0, timerCR_LEV_MAX=0.0;

	double 						timerCOMM, timerCOMM_MAX, timerCOMM_TOTAL=0.0;
	double 						timerCOMP, timerCOMP_MAX, timerCOMP_TOTAL=0.0;

	int 						dispChecks = 0;
	const int					lastRank = (int)pow(2.0,q)-1;
	cr_n = crDim;
	int world_rank;
	MPI_Comm_rank(comm, &world_rank);

	if (1){
		if ( world_rank == 0 ){
			printf(" 2.1 Starting ACR Elimination: SETUP\n\n"); fflush(stdout);
		}

		iiq = 0;
		timerCR_ALL = MPI_Wtime();
		timerCR_LEV = MPI_Wtime();

		timerCOMP = MPI_Wtime();
		MPI_print(comm, "1)     COMPUTE: Invert EVENS                                             iiq",iiq,flag_ACR_iters);
		if( world_rank % 2 == 0 ){
			inv_supermatrix(D->level[iiq]->block[world_rank]->e, acc);
		}
		MPI_Barrier(comm);
		timerCOMP = MPI_Wtime() - timerCOMP;
		MPI_Reduce(&timerCOMP, &timerCOMP_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		timerCOMP_TOTAL += timerCOMP_MAX;

		timerCOMM = MPI_Wtime();
		MPI_print(comm, "2) COMMUNICATE: EVENS_Send / ODDS_Recv D and bK from LEFT and RIGHT      iiq",iiq,flag_ACR_iters);
		if( world_rank % 2 == 0 ){
			int sendRig = world_rank+1;
			int sendLef = world_rank-1;
			// Right send
			if(1) {
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *)D->level[iiq]->block[world_rank]->e,sendRig);
				MPI_Send_HVector(comm, (HLIB::TVector *)bK->level[iiq]->block[world_rank]->e,sendRig);
			}
			// Left send
			if (sendLef>0){
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *)D->level[iiq]->block[world_rank]->e,sendLef);
				MPI_Send_HVector(comm, (HLIB::TVector *)bK->level[iiq]->block[world_rank]->e,sendLef);
			}
		}
		else if ( world_rank % 2 != 0 ){
			int recvLef = world_rank-1;
			int recvRig = world_rank+1;
			// Recv from left
			if(1) {
				D->level[iiq]->block[world_rank-1]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm, recvLef);
				bK->level[iiq]->block[world_rank-1]->e = (hlib_vector_t) MPI_Recv_Vector(comm, recvLef,block_nx);
			}
			// Recv from right
			if(recvRig<cr_n){
				D->level[iiq]->block[world_rank+1]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm, recvRig);
				bK->level[iiq]->block[world_rank+1]->e = (hlib_vector_t) MPI_Recv_Vector(comm, recvRig,block_nx);
			}
		}
		MPI_Barrier(comm);
		timerCOMM = MPI_Wtime() - timerCOMM;
		MPI_Reduce(&timerCOMM, &timerCOMM_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		timerCOMM_TOTAL += timerCOMM_MAX;

		timerCOMP = MPI_Wtime();
		MPI_print(comm, "3)     COMPUTE: ON ODDS: Compute AUXD1, AUXD2, and bK                    iiq",iiq,flag_ACR_iters);
		if( world_rank % 2 != 0 ){
			AUXD1->level[iiq+1]->block[world_rank/2]->e = hlib_matrix_copy(blankMat, NULL);
			gemm_supermatrix( 1.0, E->level[iiq]->block[world_rank]->e,          D->level[iiq]->block[world_rank-1]->e, 0.0, AUXD1->level[iiq+1]->block[world_rank/2]->e, acc);
			gemv_supermatrix(-1.0, AUXD1->level[iiq+1]->block[world_rank/2]->e, bK->level[iiq]->block[world_rank-1],   1.0, bK->level[iiq]->block[world_rank]);
			if( (world_rank+1) < cr_n ){
				AUXD2->level[iiq+1]->block[world_rank/2]->e = hlib_matrix_copy(blankMat, NULL);
				gemm_supermatrix( 1.0, F->level[iiq]->block[world_rank]->e,           D->level[iiq]->block[world_rank+1]->e, 0.0, AUXD2->level[iiq+1]->block[world_rank/2]->e, acc);
				gemv_supermatrix(-1.0, AUXD2->level[iiq+1]->block[world_rank/2]->e,  bK->level[iiq]->block[world_rank+1],   1.0, bK->level[iiq]->block[world_rank]);
			}
			copy_Vec(bK->level[iiq+1]->block[world_rank/2], bK->level[iiq]->block[world_rank]);
		}
		MPI_Barrier(comm);
		timerCOMP = MPI_Wtime() - timerCOMP;
		MPI_Reduce(&timerCOMP, &timerCOMP_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		timerCOMP_TOTAL += timerCOMP_MAX;

		if (dispTime == 1 && world_rank == 0){
			printf("    Step%2d)_timerCOMP_TOTAL = %1.6f\n", iiq, timerCOMP_TOTAL);
			printf("    Step%2d)_timerCOMM_TOTAL = %1.6f\n", iiq, timerCOMM_TOTAL);
		}

		if (dispTime == 1){
			MPI_Barrier(comm);
			timerCR_LEV = MPI_Wtime() - timerCR_LEV;
			MPI_Reduce(&timerCR_LEV, &timerCR_LEV_MAX, 1, MPI_DOUBLE,MPI_MAX, 0, comm);
			if (world_rank == 0 ) {
				printf("    __timerCR_LEV_MAX(iiq=%d): %lf\n", iiq, timerCR_LEV_MAX);
				timerCR_LEV = 0.0;
			}
		}
		cr_n = (int) (cr_n/2);

		double LOCAL_MEM_D = 0, LOCAL_MEM_E = 0, LOCAL_MEM_F = 0;
		double GLOBAL_MEM_D = 0,GLOBAL_MEM_E = 0,GLOBAL_MEM_F = 0;
		double ACCUM_ELIM_MEM = 0;

		for (iiq = 1; iiq < q; iiq++){
			timerCOMP_TOTAL = 0.0;
			timerCOMM_TOTAL = 0.0;
			timerCR_LEV_MAX = 0.0;

			timerCR_LEV = MPI_Wtime();
			timerCOMM = MPI_Wtime();
			MPI_print(comm, "4) COMMUNICATE: F,E EVENS from LEFT to ODDS                              iiq",iiq,flag_ACR_iters);
			if( oddsToInvert(q, iiq, world_rank, 0) != -1 ){
				int nextRank = (int)pow(2.0,iiq-1);
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *)F->level[iiq-1]->block[(world_rank)/nextRank]->e, world_rank+nextRank);
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *)E->level[iiq-1]->block[(world_rank)/nextRank]->e, world_rank+nextRank);
			}
			if( oddsAtLevel(q, iiq, world_rank, 0) != -1 ){ //Odds recv
				int nextRank = (int)pow(2.0,iiq-1);
				F->level[iiq-1]->block[(world_rank-nextRank)/nextRank]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm,  world_rank-nextRank ); //comp
				E->level[iiq-1]->block[(world_rank-nextRank)/nextRank]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm,  world_rank-nextRank ); //comp
			}

			MPI_print(comm, "5) COMMUNICATE: F,E EVENS from RIGHT to ODDS                             iiq",iiq,flag_ACR_iters);
			if( oddsToInvert(q, iiq, world_rank, 1) != -1 ){
				int nextRank = (int)pow(2.0,iiq-1);
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *)E->level[iiq-1]->block[(world_rank)/nextRank]->e, world_rank-nextRank); //comp
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *)F->level[iiq-1]->block[(world_rank)/nextRank]->e, world_rank-nextRank); //comp
			}
			if( oddsAtLevel(q, iiq, world_rank, 1) != -1 ){
				int nextRank = (int)pow(2.0,iiq-1);
				E->level[iiq-1]->block[(world_rank+nextRank)/nextRank]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm,  world_rank+nextRank ); //comp
				F->level[iiq-1]->block[(world_rank+nextRank)/nextRank]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm,  world_rank+nextRank ); //comp
			}
			MPI_Barrier(comm);
			timerCOMM = MPI_Wtime() - timerCOMM;
			MPI_Reduce(&timerCOMM, &timerCOMM_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
			timerCOMM_TOTAL += timerCOMM_MAX;

			LOCAL_MEM_D = 0, LOCAL_MEM_E = 0, LOCAL_MEM_F = 0;

			timerCOMP = MPI_Wtime();
			MPI_print(comm, "6)     COMPUTE: Main Schur on ODDS                                       iiq",iiq,flag_ACR_iters);
			if( oddsAtLevel(q, iiq, world_rank, 0) != -1 ){
				int nextRank = (int)pow(2.0,iiq);
				D->level[iiq]->block[world_rank/nextRank]->e = hlib_matrix_copy(blankMat, NULL); //comp
				F->level[iiq]->block[world_rank/nextRank]->e = hlib_matrix_copy(blankMat, NULL); //comp
				E->level[iiq]->block[world_rank/nextRank]->e = hlib_matrix_copy(blankMat, NULL); //comp

				if( world_rank != ((int)pow(2.0, q)-1) ){ // If not the last
					gemm_supermatrix(-1.0, AUXD1->level[iiq]->block[world_rank/nextRank]->e, F->level[iiq-1]->block[2*(world_rank/(nextRank))]->e, 1.0, D->level[iiq]->block[world_rank/nextRank]->e, acc); //comp
					gemm_supermatrix(-1.0, AUXD2->level[iiq]->block[world_rank/nextRank]->e, E->level[iiq-1]->block[2*(world_rank/(nextRank)) + 2]->e, 1.0, D->level[iiq]->block[world_rank/nextRank]->e, acc); //comp
					gemm_supermatrix(-1.0, AUXD1->level[iiq]->block[world_rank/nextRank]->e, E->level[iiq-1]->block[2*(world_rank/(nextRank))]->e, 0.0, E->level[iiq]->block[world_rank/nextRank]->e, acc); //comp
					gemm_supermatrix(-1.0, AUXD2->level[iiq]->block[world_rank/nextRank]->e, F->level[iiq-1]->block[2*(world_rank/(nextRank)) + 2]->e, 0.0, F->level[iiq]->block[world_rank/nextRank]->e, acc); //comp
				}

				else{
					gemm_supermatrix(-1.0, AUXD1->level[iiq]->block[world_rank/nextRank]->e, F->level[iiq-1]->block[2*(world_rank/(nextRank))]->e, 1.0, D->level[iiq]->block[world_rank/nextRank]->e, acc); //comp
					gemm_supermatrix(-1.0, AUXD1->level[iiq]->block[world_rank/nextRank]->e, E->level[iiq-1]->block[2*(world_rank/(nextRank))]->e, 0.0, E->level[iiq]->block[world_rank/nextRank]->e, acc); //comp
				}

				geam_supermatrix( 1.0, D->level[iiq-1]->block[world_rank/(nextRank/2)]->e, 1.0, D->level[iiq]->block[world_rank/nextRank]->e, acc); //comp
				LOCAL_MEM_E = get_Matrix_Size_MegaBytes(E->level[iiq]->block[world_rank/nextRank]->e);
				LOCAL_MEM_D = get_Matrix_Size_MegaBytes(D->level[iiq]->block[world_rank/nextRank]->e);
				LOCAL_MEM_F = get_Matrix_Size_MegaBytes(F->level[iiq]->block[world_rank/nextRank]->e);
			}

			MPI_Barrier(comm);
			MPI_Reduce(&LOCAL_MEM_E, &GLOBAL_MEM_E, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
			MPI_Reduce(&LOCAL_MEM_D, &GLOBAL_MEM_D, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
			MPI_Reduce(&LOCAL_MEM_F, &GLOBAL_MEM_F, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

			if( world_rank==0 ){
				if (dispMemory==1) printf("\n    MEM_Level_%d(E+D+F)_MB = %1.2f\n",iiq, GLOBAL_MEM_E+GLOBAL_MEM_D+GLOBAL_MEM_F); 
				ACCUM_ELIM_MEM += GLOBAL_MEM_E+GLOBAL_MEM_D+GLOBAL_MEM_F;
			}

			MPI_print(comm, "7)     COMPUTE: Inverse of evens.                                        iiq",iiq,flag_ACR_iters);
			if( oddsToInvert(q, iiq+1, world_rank, 0) != -1 ){
				int nextRank = (int)pow(2.0,iiq);
				inv_supermatrix(D->level[iiq]->block[world_rank/nextRank]->e, acc);
			}
			MPI_Barrier(comm);
			timerCOMP = MPI_Wtime() - timerCOMP;
			MPI_Reduce(&timerCOMP, &timerCOMP_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
			timerCOMP_TOTAL += timerCOMP_MAX;

			timerCOMM = MPI_Wtime();
			MPI_print(comm, "8) COMMUNICATE: EVENS LEFT D,bk to next level ODDS                       iiq",iiq,flag_ACR_iters);
			if( oddsToInvert(q, iiq+1, world_rank, 0) != -1 ){
				int nextRank = (int)pow(2.0,iiq);
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *) D->level[iiq]->block[world_rank/nextRank]->e,world_rank+nextRank); //comp
				MPI_Send_HVector(comm, (HLIB::TVector *)bK->level[iiq]->block[world_rank/nextRank]->e,world_rank+nextRank); //comp
			}
			if( oddsAtLevel(q, iiq+1, world_rank, 0) != -1 ){
				int nextRank = (int)pow(2.0,iiq);
				D->level[iiq]->block[(world_rank-nextRank)/nextRank]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm, world_rank-nextRank);
				bK->level[iiq]->block[(world_rank-nextRank)/nextRank]->e = (hlib_vector_t) MPI_Recv_Vector(comm, world_rank-nextRank,block_nx);
			}

			MPI_print(comm, "9) COMMUNICATE: EVENS RIGHT D,bk to ODDS                                 iiq",iiq,flag_ACR_iters);
			if( oddsToInvert(q, iiq+1, world_rank, 1) != -1 ){
				int nextRank = (int)pow(2.0,iiq);
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *) D->level[iiq]->block[world_rank/nextRank]->e,world_rank-nextRank); //comp
				MPI_Send_HVector(comm, (HLIB::TVector *)bK->level[iiq]->block[world_rank/nextRank]->e,world_rank-nextRank); //comp
			}
			if( oddsAtLevel(q, iiq+1, world_rank, 1) != -1 ){
				int nextRank = (int)pow(2.0,iiq);
				 D->level[iiq]->block[(world_rank+nextRank)/nextRank]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm, world_rank+nextRank); //comp
				bK->level[iiq]->block[(world_rank+nextRank)/nextRank]->e = (hlib_vector_t) MPI_Recv_Vector(comm, world_rank+nextRank,block_nx); //comp
			}
			MPI_Barrier(comm);
			timerCOMM = MPI_Wtime() - timerCOMM;
			MPI_Reduce(&timerCOMM, &timerCOMM_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
			timerCOMM_TOTAL += timerCOMM_MAX;

			timerCOMP = MPI_Wtime();
			MPI_print(comm, "10)    COMPUTE: AUXD1, AUXD2 and bK ON ODDS                              iiq",iiq,flag_ACR_iters);
			if( oddsAtLevel(q, iiq+1, world_rank, 0) != -1 ){
				int nextRank = (int)pow(2.0,iiq+1);
				int consecRank = (world_rank)/nextRank;
				int evenRank = consecRank*2;
				int oddRank = (world_rank)/(nextRank/2);
				int NextEvenRank = evenRank+2;
				if (1){
					AUXD1->level[iiq+1]->block[consecRank]->e = hlib_matrix_copy(blankMat, NULL);
					gemm_supermatrix( 1.0, E->level[iiq]->block[oddRank]->e, D->level[iiq]->block[evenRank]->e, 0.0, AUXD1->level[iiq+1]->block[consecRank]->e, acc);
					gemv_supermatrix(-1.0, AUXD1->level[iiq+1]->block[consecRank]->e,bK->level[iiq]->block[evenRank], 1.0, bK->level[iiq]->block[oddRank]);
				}
				if( world_rank != (int) pow(2.0 , q) - 1 ){ //if not the last
					AUXD2->level[iiq+1]->block[consecRank]->e = hlib_matrix_copy(blankMat, NULL);
					gemm_supermatrix( 1.0,     F->level[iiq]->block[oddRank]->e, D->level[iiq]->block[NextEvenRank]->e, 0.0, AUXD2->level[iiq+1]->block[consecRank]->e, acc);
					gemv_supermatrix(-1.0, AUXD2->level[iiq+1]->block[consecRank]->e, bK->level[iiq]->block[NextEvenRank], 1.0, bK->level[iiq]->block[oddRank]);
				}
				copy_Vec(bK->level[iiq+1]->block[consecRank], bK->level[iiq]->block[oddRank]);
			}
			MPI_Barrier(comm);
			timerCOMP = MPI_Wtime() - timerCOMP;
			MPI_Reduce(&timerCOMP, &timerCOMP_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
			timerCOMP_TOTAL += timerCOMP_MAX;

			if (dispChecks == 1){
				if(world_rank==3){
					printf("Step 10) - Correctness Check\n");
					probe_HVector(comm, bK->level[1]->block[1]->e, world_rank, "bK",1,1);
					probe_HVector(comm, bK->level[2]->block[0]->e, world_rank, "bK",2,0);
				}
				if(world_rank==7){
					probe_HVector(comm, bK->level[1]->block[3]->e, world_rank, "bK",1,3);
					probe_HVector(comm, bK->level[2]->block[1]->e, world_rank, "bK",2,1);
				}
			}
			
			if (dispTime == 1 && world_rank == 0){
				printf("    Step%2d)_timerCOMP_TOTAL = %1.6f\n", iiq, timerCOMP_TOTAL);
				printf("    Step%2d)_timerCOMM_TOTAL = %1.6f\n", iiq, timerCOMM_TOTAL);
			}

			if (dispTime == 1){
				MPI_Barrier(comm);
				timerCR_LEV = MPI_Wtime() - timerCR_LEV;
				MPI_Reduce(&timerCR_LEV, &timerCR_LEV_MAX, 1, MPI_DOUBLE,MPI_MAX, 0, comm);
				if (world_rank == 0 ) {
					printf("    __timerCR_LEV_MAX(iiq=%d): %lf\n", iiq, timerCR_LEV_MAX);
				}
			}

			MPI_Barrier(comm);
			cr_n = (int) (cr_n/2);
		} // iiq loop (end)

		timerCR_LEV = MPI_Wtime();
		iiqK = q-1;
		timerCOMP_TOTAL = 0.0;
		timerCOMM_TOTAL = 0.0;

		timerCOMM = MPI_Wtime();
		MPI_print(comm, "1)  COMMUNICATION: F to last             iiqK",iiqK,flag_ACR_iters);
		if( oddsAtLevel(q, iiqK, world_rank, 0) != -1 ){
			if ( world_rank == oddsAtLevel(q, iiqK, world_rank, -1) ){ //last
				int first = (int) pow(2.0 , iiqK) - 1;
				F->level[iiq-1]->block[0]->e = (hlib_matrix_t) MPI_Recv_Matrix(comm, first);
			}
			else{ //first
				int last = (int) pow(2.0 , q) - 1;
				MPI_Send_HMatrix(comm, (HLIB::TMatrix *) F->level[iiq-1]->block[0]->e, last);
			}
		}
		MPI_Barrier(comm);

		timerCOMM = MPI_Wtime() - timerCOMM;
		MPI_Reduce(&timerCOMM, &timerCOMM_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		timerCOMM_TOTAL += timerCOMM_MAX;

		timerCOMP = MPI_Wtime();
		MPI_print(comm, "2)    COMPUTATION: First part of the solution on R_last          iiqK",iiq,flag_ACR_iters);
		if (world_rank == (int)pow(2.0,q)-1){ //last
			D->level[q]->block[0]->e = hlib_matrix_copy(blankMat, NULL); //comp
			hlib_matrix_t P1 = hlib_matrix_copy(blankMat, NULL); //comp
			gemm_supermatrix( 1.0, D->level[q-1]->block[0]->e, F->level[q-1]->block[0]->e, 0.0, P1, acc); //comp
			gemm_supermatrix(-1.0, E->level[q-1]->block[1]->e, P1, 1.0, D->level[q]->block[0]->e, acc); //comp
			geam_supermatrix( 1.0, D->level[q-1]->block[1]->e, 1.0, D->level[q]->block[0]->e, acc); //comp
			inv_supermatrix(D->level[q]->block[0]->e, acc); //comp
		}
		MPI_Barrier(comm);
		timerCOMP = MPI_Wtime() - timerCOMP;
		MPI_Reduce(&timerCOMP, &timerCOMP_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		timerCOMP_TOTAL += timerCOMP_MAX;

		LOCAL_MEM_D = 0; GLOBAL_MEM_D = 0;
		if ( world_rank == lastRank ){
			LOCAL_MEM_D = get_Matrix_Size_MegaBytes(D->level[q]->block[0]->e);
			if (dispMemory==1) printf("\n    MEM_Level_%d(0+D+0)_MB = %1.2f\n",iiq, LOCAL_MEM_D); 
		}

		MPI_Barrier(comm);
		MPI_Reduce(&LOCAL_MEM_D, &GLOBAL_MEM_D, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

		if ( world_rank==0 ){
			ACCUM_ELIM_MEM += GLOBAL_MEM_D;
			*ELIM_MEM = ACCUM_ELIM_MEM;
		}
		
		if (dispTime == 1 && world_rank == 0){
			printf("    Step%2d)_timerCOMP_TOTAL = %1.6f\n", iiq, timerCOMP_TOTAL);
			printf("    Step%2d)_timerCOMM_TOTAL = %1.6f\n", iiq, timerCOMM_TOTAL);
		}

		if (dispTime == 1){
			MPI_Barrier(comm);
			timerCR_LEV = MPI_Wtime() - timerCR_LEV;
			MPI_Reduce(&timerCR_LEV, &timerCR_LEV_MAX, 1, MPI_DOUBLE,MPI_MAX, 0, comm);
			if (world_rank == 0 ) {
				printf("    __timerCR_LEV_MAX(iiq=%d): %lf\n", iiq, timerCR_LEV_MAX);
			}
		}

		MPI_Barrier(comm);
		timerCR_ALL = MPI_Wtime() - timerCR_ALL;

		if (dispTime == 1){
			MPI_Reduce(&timerCR_ALL, &timerCR_ALL_MAX, 1, MPI_DOUBLE,MPI_MAX, 0, comm);
			if (world_rank == 0 ) {
				printf("\n");
				printf("   ━━━timerCR_ALL_MAX: %lf\n\n", timerCR_ALL_MAX);
			}
		}
	}
}

double
acr_apply(MPI_Comm comm, pcrDiag E, pcrDiag D, pcrDiag F, pcrVec bK, int q, int crDim, int block_nx,
 pcrVec levSol, pcrVecLevel gatherSol, int flag_ACR_iters, int dispTime, hlib_permutation_t perm){
	int 		globalIdx;
	int 		destination1;
	int 		sendingRank;
	int 		iiq, iiqK;
	int 		cr_n = crDim;
	int 		destinationZERO = 0;
	double 		timerBS_ALL=0.0, timerBS_ALL_MAX=0.0;
	const int	lastRank = (int)pow(2.0,q)-1;
	pVec        v1,v2,v3;

	v1 = new_Vec(block_nx);
	v2 = new_Vec(block_nx);
	v3 = new_Vec(block_nx);

	int world_rank;
	MPI_Comm_rank(comm, &world_rank);

	if ( world_rank == 0 && flag_ACR_iters == 1) {
		printf("\n 2.2 Starting ACR Back-substitution \n");
	}

	MPI_Barrier(comm);
	timerBS_ALL = MPI_Wtime();

	iiqK = q-1;
	if ( world_rank == lastRank ){
		gemv_supermatrix(1.0, D->level[q]->block[0]->e, bK->level[q]->block[0], 0.0, levSol->level[q]->block[0]);
		globalIdx = crDim-1; //Last block
		MPI_Send_HVector(comm, (HLIB::TVector *)levSol->level[q]->block[0]->e, destinationZERO);
		copy_Vec(gatherSol->block[globalIdx], levSol->level[q]->block[0]);
		hlib_vector_permute(gatherSol->block[globalIdx]->e, perm, NULL);
	}

	// Gather RECV
	if (world_rank==0){
		sendingRank = crDim-1;
		globalIdx = crDim-1;
		gatherSol->block[globalIdx]->e = (hlib_vector_t) MPI_Recv_Vector(comm, sendingRank,block_nx);
		hlib_vector_permute(gatherSol->block[globalIdx]->e, perm, NULL);
	}

	MPI_print(comm, "3)    COMPUTATION: Back-substitution loop                    iiqK",iiqK,flag_ACR_iters);
	for (iiq = 0; iiq < q; iiq++){
		
		int skipSize = (int)pow(2.0,iiqK);
		int skipSizet2 = skipSize*2;

		// Computation and Gather SEND v2
		if ( oddsAtLevel(q, iiqK+1, world_rank, 0) != -1 ){
			int nextRank = (int)pow(2.0,iiqK+1);
			int consecRank = world_rank/nextRank;
			int evenRank = consecRank*2;
			int prevConsecRank = consecRank-1;
			int permSoln = consecRank;
			int permSolnBuffer = permSoln;

			if ( world_rank == oddsAtLevel(q, iiqK+1, world_rank, -1) ){
				scale_Vec(v3, 0.0);					
				gemv_supermatrix(1.0, E->level[iiqK]->block[evenRank]->e, levSol->level[iiqK+1]->block[prevConsecRank], 0.0, v1);
				gemv_supermatrix(1.0, F->level[iiqK]->block[evenRank]->e, levSol->level[iiqK+1]->block[consecRank], 0.0, v2);
				add_Vec(v3, -1.0, v1);
				add_Vec(v3, -1.0, v2);
				add_Vec(v3, 1.0, bK->level[iiqK]->block[evenRank]);
				gemv_supermatrix(1.0, D->level[iiqK]->block[evenRank]->e, v3, 0.0, levSol->level[iiqK]->block[permSolnBuffer*2]);
				
				// Gather solution vector
				globalIdx = (int)pow(2.0,iiqK)+consecRank*skipSizet2-1;
				MPI_Send_HVector(comm, (HLIB::TVector *)levSol->level[iiqK]->block[permSolnBuffer*2]->e, destinationZERO); //GatherSEND
				MPI_Send_HVector(comm, (HLIB::TVector *)levSol->level[iiqK]->block[permSolnBuffer*2]->e, globalIdx); //GatherSEND
			}
			// Computation & Gather SEND
			else {
				gemv_supermatrix(1.0, F->level[iiqK]->block[evenRank]->e, levSol->level[iiqK+1]->block[consecRank],0.0,v1);
				scale_Vec(v3, 0.0);
				add_Vec(v3, -1.0, v1);
				add_Vec(v3, 1.0, bK->level[iiqK]->block[evenRank]);
				gemv_supermatrix(1.0, D->level[iiqK]->block[evenRank]->e, v3, 0.0, levSol->level[iiqK]->block[permSolnBuffer*2]);
				globalIdx = (int)pow(2.0,iiqK)+consecRank*skipSizet2-1;
				MPI_Send_HVector(comm, (HLIB::TVector *)levSol->level[iiqK]->block[permSolnBuffer*2]->e, destinationZERO); //GatherSEND
				MPI_Send_HVector(comm, (HLIB::TVector *)levSol->level[iiqK]->block[permSolnBuffer*2]->e, globalIdx); //GatherSEND
			}

			copy_Vec(levSol->level[iiqK]->block[permSolnBuffer*2+1], levSol->level[iiqK+1]->block[consecRank]); //comp
		}

		// Gather RECV  v2: To rank 0
		if ( world_rank == 0){
			int iloop;
			for (iloop = 0; iloop < crDim; iloop++) {
				if ( oddsAtLevel(q, iiqK+1, iloop, 0) != -1 ){
					int nextRank = (int)pow(2.0,iiqK+1);
					int consecRank = iloop/nextRank;
					globalIdx = (int)pow(2.0,iiqK)+consecRank*skipSizet2-1;
					sendingRank = iloop;
					gatherSol->block[globalIdx]->e = (hlib_vector_t) MPI_Recv_Vector(comm, sendingRank,block_nx);
					hlib_vector_permute(gatherSol->block[globalIdx]->e, perm, NULL); //Permute solution back to natural ordering
				}
			}
		}

		// RECV_GatherN to rank globalIdx
		if ( ranksToScatter(q, iiqK, world_rank) != -1 ){
			globalIdx = (int)pow(2.0,iiqK)+world_rank;
			gatherSol->block[world_rank]->e = (hlib_vector_t) MPI_Recv_Vector(comm, globalIdx,block_nx);
			hlib_vector_permute(gatherSol->block[world_rank]->e, perm, NULL); //Permute solution back to natural ordering
		}

		// Communication SEND: V2
		if ( oddsAtLevel(q, iiqK+1, world_rank, 0) != -1){
			int nextRank = (int)pow(2.0,iiqK+1);
			int consecRank = world_rank/nextRank;
			int permSoln = consecRank;
			int permSolnBuffer = permSoln;
			if ( iiq < q-1 ){
				destination1 = world_rank-skipSize;
				if (world_rank != destination1){
					MPI_Send_HVector(comm, (HLIB::TVector *)levSol->level[iiqK]->block[permSolnBuffer*2]->e,destination1);
				} 
				destination1 = world_rank+skipSize;
				if (world_rank != destination1 && destination1 < cr_n){
					MPI_Send_HVector(comm, (HLIB::TVector *)levSol->level[iiqK]->block[permSolnBuffer*2+1]->e,destination1);
				}
			}
		}

		if( oddsAtLevel(q, iiqK, world_rank, 0) != -1 && iiq<(q-1)){
			int nextRank = (int)pow(2.0,iiqK+1);
			int consecRank = world_rank/nextRank;
			int destination1;
			int permSoln = consecRank;
			int permSolnBuffer = permSoln;
			if ( (world_rank/skipSize) % 2 == 0){
				destination1 = world_rank+skipSize;
				int idx1 = permSolnBuffer*2;
				levSol->level[iiqK]->block[idx1]->e = (hlib_vector_t) MPI_Recv_Vector(comm, destination1,block_nx);
			}
			int firstRank = (int)pow(2.0, q-iiq-1)-1;
			if ( (world_rank/skipSize) % 2 == 0  && (world_rank!=firstRank) ){
				destination1 = world_rank-skipSize;
				int idx1 = permSolnBuffer*2-1;
				levSol->level[iiqK]->block[idx1]->e = (hlib_vector_t) MPI_Recv_Vector(comm, destination1,block_nx);
			}
		}

		iiqK = iiqK - 1;
		MPI_Barrier(comm);
	}
	
	MPI_Barrier(comm);
	timerBS_ALL = MPI_Wtime() - timerBS_ALL;
	MPI_Reduce(&timerBS_ALL, &timerBS_ALL_MAX, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	if (world_rank == 0 && dispTime == 1) {
		printf("\n   ━━━timerBS_ALL_MAX: %lf\n\n", timerBS_ALL_MAX);
	}
	return timerBS_ALL_MAX;
}