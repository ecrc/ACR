#include "acr_lib.h"

int main ( int argc, char ** argv ){
    
    // H-matrices
    int                         info;
	double                      heps;
    double                      eta;
	hlib_permutation_t 			perm;
	hlib_permutation_t 			perm_e2i;
    int                         TBB_THREADS;

	// Problem size
	int                         hDim, crDim, nmin, q, nx, N, block_nx;
	double                      h;

    // ACR Variables
    pcrDiag                     E,D,F;
	pcrDiag    					AUXD1,AUXD2;
    pcrVec                      bK;
    
    // Files
    char                        *fblank;
    char                        folder[250];
    char                        buffer_rhs[250];
    char                        buffer_sol[250];
    char                        buffer_mySpa[250];
    char                        buffer_blank[250];
	
    // Memory
    double 						SETUP_MEM_MB		= 0;
    double 						ELIM_MEM_MB 		= 0;
    double 						TOTAL_MEM_MB		= 0;

	// Profiling
    int                         flag_ACR_iters 	 	= 0;
    int 						flag_verbose_ACR 	= 1;
	int 						checkError 			= 1;

    // Plotting matrices
	int 						visualizeEPS 		= 1;
	int 						generateEPS			= 0;
	int 						generateFirstBlock	= 0;
	// 1 = Heat map. 2 = SVD. 3 = Colors only (1 for rank mining)
	int 						epsTYPE 			= 3; 
	
    // Statistic of memory and time
	int 						dispMemory 			= 1;
	int 						dispTime 			= 1;
	int 						dispSol				= 1;
    hlib_acc_t                  acc;

	MPI_Init(&argc, &argv);
	MPI_Comm comm; 
	comm = MPI_COMM_WORLD;
	int world_rank;
	MPI_Comm_rank(comm, &world_rank);

    // Parsing command line
    if (argc == 7){
    	TBB_THREADS 	= atoi( argv[1] );
        hDim 			= atoi( argv[2] );
        crDim 			= atoi( argv[3] );
        nmin 			= atoi( argv[4] );
        eta 			= (double) atof ( argv[5] );
        heps 			= (double) atof ( argv[6] );
        q = (int) ( log( (double)crDim ) / log(2.0) );
	    nx = hDim;
	    h = 1.0 / ( (double) (nx - 1) );
        sprintf(folder, "../inputFolder/");
	}
    else {
        printf( " \n\n ━━ ERROR: Needs 6 input parameters (SHARED_MEM_CORES hDIM crDIM NMIN ETA Heps). Aborting execution\n");
        return 0;
    }

    // 0. Parameters setting
    if ( 1 ){
		hlib_init( &info );
		hlib_set_verbosity( 0 );
		sprintf(buffer_rhs, "%srhs_0.mat",folder);
		sprintf(buffer_sol, "%ssol_0.mat",folder);
		sprintf(buffer_mySpa, "%sK3D_0.mat",folder);
		sprintf(buffer_blank, "%sD0.mat",folder);
		fblank = buffer_blank;
		block_nx = nx*nx;
	    N = block_nx * crDim;
		acc = hlib_acc_fixed_eps( heps );
		hlib_set_nthreads(TBB_THREADS);
		hlib_set_coarsening(1, 0);
		hlib_set_recompress(1);
	}

	if(world_rank==0){
	    printf("=====================================================================\n");
	    printf(" # ACR        N = (%dx%dx%d) = %d     folder = %s\n",nx,nx,crDim,N,folder);
	    printf(" # TBB = %-2d   nmin = %d   eta = %1.2f   H_eps = %1.0e\n",TBB_THREADS,nmin,eta,heps);
	    printf("=====================================================================\n"); fflush(stdout);
	}
	
	if(world_rank==0 && flag_verbose_ACR == 1){	printf("1. Loading and setup\n"); }

	D = new_crDiag(q,block_nx);
	E = new_crDiag(q,block_nx);
	F = new_crDiag(q,block_nx);
	hlib_blockclustertree_t     bct;
	hlib_matrix_t               blankMat = NULL;
	blankMat = set_blankMat(fblank, nx, h, eta, nmin, &bct, acc);
	perm = hlib_matrix_row_perm_i2e(blankMat, &info);
	perm_e2i = hlib_matrix_row_perm_e2i(blankMat, &info);
    set_A0(comm, crDim, E, D, F, acc, blankMat, bct, folder, dispMemory, &SETUP_MEM_MB);
	bK = new_crVec(q,block_nx);
	set_RHS(comm, folder, perm_e2i, bK);
	AUXD1 = new_crDiag(q,block_nx);
	AUXD2 = new_crDiag(q,block_nx);

    // 2. and 3. Solve and Error checking
    if ( 1 ){
		pcrVecLevel gatherSol = new_crVecLevel(q+1, 0, block_nx); //Creates a new_crVec just at level iiq 0 in all ranks
		pcrVec levSol;
		levSol = new_crVec(q,block_nx);
		solve_ACR_parallel(comm, E, D, F, AUXD1, AUXD2, blankMat, bK, levSol, gatherSol, q, crDim, block_nx, acc, &ELIM_MEM_MB, flag_ACR_iters, dispMemory, dispTime, perm);

		// 3. Error checking
		if ( world_rank==0 && checkError == 1 ){

			if(flag_verbose_ACR == 1){
				printf("3. Residual norm checking\n");
			}

			hlib_matrix_t   K2D = NULL;
		    char 			fileName_A[120];
			sprintf(fileName_A, "%sK3D_0.mat",folder);
			printf("%s\n",fileName_A);
			K2D = hlib_load_matrix( fileName_A, NULL );

			char file_fullEXACT[250];
			pVec fullEXACT = new_Vec(N);
			sprintf(file_fullEXACT, "%s_Full_sol_0.mat", folder);
			fullEXACT->e = hlib_load_vector( file_fullEXACT, NULL );

			pVec fullRHS = new_Vec(N);
			double norm2_fullRHS;
			char file_fullRHS[250];
			sprintf(file_fullRHS, "%s_Full_rhs_0.mat", folder);
			fullRHS->e = hlib_load_vector( file_fullRHS, NULL );
			norm2_fullRHS = hlib_vector_norm2( fullRHS->e, NULL );

			printf("\n");
			pVec fullSolution = new_Vec(N);
			int CRloop, BLOCKloop;
			double entryACR, entryEXACT;
			int counter = 0;
			for (CRloop = 0; CRloop < crDim; CRloop++){
				for (BLOCKloop = 0; BLOCKloop < block_nx; BLOCKloop++){
					entryEXACT 	= hlib_vector_entry_get(fullEXACT->e, counter, NULL); //Exact solution
					entryACR 	= hlib_vector_entry_get(gatherSol->block[CRloop]->e, BLOCKloop, NULL); //ACR solution
					hlib_vector_entry_set(fullSolution->e, counter, entryACR, NULL);
					counter++;
				}
			}

			pVec fullResidual = new_Vec(N);
			gemv_supermatrix(1.0, K2D, fullSolution, 0.0, fullResidual);
			int fullSolution_loop;
			double residualEntry;
			double rhsEntry;

			for (fullSolution_loop = 0; fullSolution_loop < N; fullSolution_loop++) {
				residualEntry = hlib_vector_entry_get(fullResidual->e, fullSolution_loop, NULL);
				rhsEntry = hlib_vector_entry_get(fullRHS->e, fullSolution_loop, NULL);
				hlib_vector_entry_set(fullResidual->e, fullSolution_loop, residualEntry-rhsEntry, NULL);
			}

			double norm2_fullResidual;
			norm2_fullResidual = hlib_vector_norm2( fullResidual->e, NULL );
			printf("||Ax-b||/||b| = %1.2e\n", norm2_fullResidual/norm2_fullRHS );
		}
	}

	// 4. Processing ranks
	if ( generateEPS == 1 ){

		if ( world_rank == 0){
			printf("4. Processing ranks\n");
		}

		TPSMatrixVis mvis;
		// Getting first block of ACR
		if ( world_rank == 0 && generateFirstBlock == 1){
			char *NAME = "Df";
			char NAME_AND_EXTENSION[124];
			char NAME_AND_COMMAND_VISUALIZE[255];
			char NAME_AND_COMMAND_MAX_RANKS[255];
			sprintf(NAME_AND_EXTENSION,"%s_%dc_eta_%d_t%d.ps",NAME,nx,(int)eta,epsTYPE);
			sprintf(NAME_AND_COMMAND_VISUALIZE,"xdg-open %s > /dev/null 2>&1 &",NAME_AND_EXTENSION);
			sprintf(NAME_AND_COMMAND_MAX_RANKS,"grep -Po '(?<=\\().*?(?=\\))' %s | sort -nur | tr '\n' ' '",NAME_AND_EXTENSION);

			switch (epsTYPE){
				case 1: mvis.structure( true ).rank_col( true );  break;
				case 2: mvis.structure( false ).rank_col( true ); break;
				case 3: mvis.svd( true );  break;
			}
			mvis.print( (HLIB::TMatrix *)D->level[0]->block[2]->e, NAME_AND_EXTENSION );
			if ( visualizeEPS == 1 ) system(NAME_AND_COMMAND_VISUALIZE);
			printf(" Ranks_FIRST_block: ");
			system(NAME_AND_COMMAND_MAX_RANKS);
			printf("\n");
		}

		MPI_Barrier(comm);
		// Getting last block of ACR
		int lastRank = (int)pow(2.0,q)-1;
		if ( world_rank == lastRank){
			char *NAME = "Dl";
			char NAME_AND_EXTENSION[124];
			char NAME_AND_COMMAND_VISUALIZE[255];
			char NAME_AND_COMMAND_MAX_RANKS[255];
			sprintf(NAME_AND_EXTENSION,"%s_%dc_eta_%d_t%d.ps",NAME,nx,(int)eta,epsTYPE);
			sprintf(NAME_AND_COMMAND_VISUALIZE,"xdg-open %s > /dev/null 2>&1 &",NAME_AND_EXTENSION);
			sprintf(NAME_AND_COMMAND_MAX_RANKS,"grep -Po '(?<=\\().*?(?=\\))' %s | sort -nur | tr '\n' ' '",NAME_AND_EXTENSION);

			switch (epsTYPE){
				case 1: mvis.structure( true ).rank_col( true );  break;
				case 2: mvis.svd( true ).svd_ref( norm_2( (HLIB::TMatrix *)D->level[q]->block[0]->e ) );  break;
				case 3: mvis.svd( true ); break;
				case 4: mvis.structure( false ).rank_col( true ); break;
			}

			mvis.print( (HLIB::TMatrix *)D->level[q-1]->block[0]->e, NAME_AND_EXTENSION );
			if ( visualizeEPS == 1 ) system(NAME_AND_COMMAND_VISUALIZE);
			printf("\n Ranks_LAST_block: ");
			system(NAME_AND_COMMAND_MAX_RANKS);
			printf("\n");
		}
		MPI_Barrier(comm);
	}

	// 5. Memory check
	if ( 1 ){
		if ( world_rank==0 ){
			printf("\n5. Memory check\n");
			TOTAL_MEM_MB = SETUP_MEM_MB + ELIM_MEM_MB;
			printf(" TOTAL_MEM_MB = %1.2f\n\n",TOTAL_MEM_MB);
		}
	}

	hlib_done( &info );
	MPI_Finalize();
    return 0;
}