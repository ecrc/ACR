#include "acr_mpi.h"
#include "matrix/TBSHMBuilder.hh"
#include "misc/TByteStream.hh"
using namespace HLIB;

void
MPI_print(MPI_Comm comm, char const *MESSAGE, int num, int verbose){
	if (verbose != 0){
		int world_rank;
		MPI_Comm_rank(comm, &world_rank);
		if(world_rank == 0){
			printf("   %s = %d\n",MESSAGE,num); fflush(stdout);
		}
	}
}

void
MPI_blank_line(MPI_Comm comm){
	int world_rank;
	MPI_Comm_rank(comm, &world_rank);
	if(world_rank == 0){
		printf("\n"); fflush(stdout);
	}
}

void
probe_HMatrix(MPI_Comm comm, hlib_matrix_t A, int MPI_rank, char const *NAME, int level, int block){
	int world_rank;	
	MPI_Comm_rank(comm, &world_rank);
	if( world_rank == MPI_rank ){
		printf("[At R%-2d] %s[%d][%d]_(0,0) = %f\n",world_rank, NAME, level, block, hlib_matrix_entry_get( A, 0, 0, NULL));
	}
}

void
probe_HVector(MPI_Comm comm, hlib_vector_t vec, int MPI_rank, char const *NAME, int level, int block){
	int world_rank;	
	MPI_Comm_rank(comm, &world_rank);
	if( world_rank == MPI_rank ){
		printf("[At R%-2d] %s[%-2d][%-2d]_(0) = %f\n", world_rank, NAME, level, block, hlib_vector_entry_get( vec, 0, NULL) );
	}
}

void // Messages up to 2GB
MPI_Send_HMatrix_1m(MPI_Comm comm, TMatrix* A, int destination){
	int tag = 0;
	TByteStream bs;
	size_t size_bs = A->bs_size();
	bs.set_size( size_bs );
	A->write( bs );
	MPI_Send(bs.data(), bs.size(), MPI_BYTE, destination, tag, comm);
} 

TMatrix* // Messages up to 2GB
MPI_Recv_Matrix_1m(MPI_Comm comm, int source){
	MPI_Status status;
	int BS_SIZE;
	int tag = 0;
	MPI_Probe( source, tag , comm, &status);
	MPI_Get_count( &status, MPI_BYTE, &BS_SIZE );
	TByteStream bs;
	size_t size_bs = BS_SIZE;
	bs.set_size( size_bs );
	MPI_Recv(bs.data(), bs.size(), MPI_BYTE, source, tag, comm, &status);
	
	TBSHMBuilder builder;
	TMatrix* A = builder.build( bs ); 
	return A;
} 

void // Messages up to 4GB
MPI_Send_HMatrix_2m(MPI_Comm comm, TMatrix* A, int destination){
	
	TByteStream bs;
	size_t size_bs = A->bs_size();
	bs.set_size( size_bs );
	A->write( bs );

	if (size_bs <= INT_MAX){
		printf("Send (1m)\n");
		MPI_Send(bs.data(), bs.size(), MPI_BYTE, destination, 0 /*tag*/, comm);
	}

	else {
		printf("Send (2m)\n");
		
		int blocksCount = 3;
		int blocksLength[3] = {1, 1, 1};
		MPI_Aint offsets[3];
		MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
		MPI_Datatype type_dsi2m;
		offsets[0] = offsetof(dsi2m, middleIdx);
		offsets[1] = offsetof(dsi2m, sizeFH);
		offsets[2] = offsetof(dsi2m, sizeSH);
		MPI_Type_create_struct(blocksCount, blocksLength, offsets, types, &type_dsi2m);
		MPI_Type_commit(&type_dsi2m);
		uchar *data = bs.data();

		dsi2m send;
		send.middleIdx 		= size_bs/2;
		send.sizeFH 		= send.middleIdx;
		send.sizeSH 		= size_bs - send.middleIdx;

		MPI_Send(&send, 1, type_dsi2m, destination /*rank*/, 0 /*tag*/, comm);
		MPI_Send(&data[0], 				send.sizeFH, MPI_UNSIGNED_CHAR, destination, 10 /*tag*/, comm);
		MPI_Send(&data[send.middleIdx], send.sizeSH, MPI_UNSIGNED_CHAR, destination, 11 /*tag*/, comm);
	}
} 

TMatrix*  // Messages up to 4GB
MPI_Recv_Matrix_2m(MPI_Comm comm, int source){
	MPI_Status status;
	int BS_SIZE;
	int tag = 0;
	MPI_Probe( source, MPI_ANY_TAG, comm, &status);
	MPI_Get_count( &status, MPI_BYTE, &BS_SIZE );

	TByteStream bs;
	size_t size_bs = BS_SIZE;
	bs.set_size( size_bs );

	// Message size is less than INT_MAX, but not the datastructure for the size of the split message
	if ( BS_SIZE <= INT_MAX && BS_SIZE > 1024){
		printf("Recv (1m)\n");
		int	errorCode = MPI_Recv(bs.data(), bs.size(), MPI_BYTE, source, tag, comm, &status);
		if( errorCode != MPI_SUCCESS /*errorCode != 0 */){
			printf("GC: MPI_Recv_Matrix: R%d received %d chars, from R%d, err = %d\n",
			(int)source, (int)bs.size(), (int)status.MPI_SOURCE, (int)status.MPI_ERROR);
		}
	}

	else{
		printf("Recv (2m)\n");
		int blocksCount = 3;
		int blocksLength[3] = {1, 1, 1};
		MPI_Aint offsets[3];
		MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
		MPI_Datatype type_dsi2m;
		offsets[0] = offsetof(dsi2m, middleIdx);
		offsets[1] = offsetof(dsi2m, sizeFH);
		offsets[2] = offsetof(dsi2m, sizeSH);
		MPI_Type_create_struct(blocksCount, blocksLength, offsets, types, &type_dsi2m);
		MPI_Type_commit(&type_dsi2m);

		dsi2m recv;
		int	errorCode = MPI_Recv(&recv, 1, type_dsi2m, source /*rank*/, 0 /*tag*/, comm, &status);

		size_t size_bs = (size_t)recv.sizeFH+(size_t)recv.sizeSH;
		uchar *data = (uchar *) malloc( size_bs );

		errorCode += MPI_Recv(&data[0], 				recv.sizeFH, MPI_UNSIGNED_CHAR, source, 10 /*tag*/, comm , &status);
		errorCode += MPI_Recv(&data[recv.middleIdx],	recv.sizeSH, MPI_UNSIGNED_CHAR, source, 11 /*tag*/, comm , &status);

		if( errorCode != MPI_SUCCESS /*errorCode != 0 */){
			printf("GC: MPI_Recv_Matrix: R%d received %d chars, from R%d, err = %d\n",
			(int)source, (int)bs.size(), (int)status.MPI_SOURCE, (int)status.MPI_ERROR);
		}
		bs.copy_stream( data, size_bs);
		bs.set_pos( 0 );
	}

	TBSHMBuilder builder;
	TMatrix* A = builder.build( bs );
	return A;
}

void  // Messages up to 8GB
MPI_Send_HMatrix(MPI_Comm comm, TMatrix* A, int destination){
	
	TByteStream bs;
	size_t size_bs = A->bs_size();
	bs.set_size( size_bs );
	A->write( bs );

	if (size_bs <= INT_MAX){
		MPI_Send(bs.data(), bs.size(), MPI_BYTE, destination, 0 /*tag*/, comm);
	}

	else {
		printf("Send (4m)\n");

		int blocksCount = 4;
		int blocksLength[4] = {1, 1, 1, 1};
		MPI_Aint offsets[4];
		MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
		MPI_Datatype type_dsi4m;
		offsets[0] = offsetof(dsi4m, sizem1);
		offsets[1] = offsetof(dsi4m, sizem2);
		offsets[2] = offsetof(dsi4m, sizem3);
		offsets[3] = offsetof(dsi4m, sizem4);
		MPI_Type_create_struct(blocksCount, blocksLength, offsets, types, &type_dsi4m);
		MPI_Type_commit(&type_dsi4m);
		uchar *data = bs.data();

		dsi4m send;
		send.sizem1	= size_bs/4;
		send.sizem2	= send.sizem1;
		send.sizem3	= send.sizem1;
		send.sizem4	= size_bs - (3*send.sizem1);
		MPI_Send(&send, 1, type_dsi4m, destination /*rank*/, 0 /*tag*/, comm);

		size_t idx1 = (size_t)1*(size_t)send.sizem1;
		size_t idx2 = (size_t)2*(size_t)send.sizem1;
		size_t idx3 = (size_t)3*(size_t)send.sizem1;

		MPI_Send(&data[0],	 	send.sizem1, MPI_UNSIGNED_CHAR, destination, 11 /*tag*/, comm);
		MPI_Send(&data[idx1],	send.sizem2, MPI_UNSIGNED_CHAR, destination, 12 /*tag*/, comm);
		MPI_Send(&data[idx2],	send.sizem3, MPI_UNSIGNED_CHAR, destination, 13 /*tag*/, comm);
		MPI_Send(&data[idx3],	send.sizem4, MPI_UNSIGNED_CHAR, destination, 14 /*tag*/, comm);
	}
} 

TMatrix* // Messages up to 8GB
MPI_Recv_Matrix(MPI_Comm comm, int source){
	MPI_Status status;
	int BS_SIZE;
	int tag = 0;
	MPI_Probe( source, MPI_ANY_TAG, comm, &status);
	MPI_Get_count( &status, MPI_BYTE, &BS_SIZE );

	TByteStream bs;
	size_t size_bs = BS_SIZE;
	bs.set_size( size_bs );

	if ( BS_SIZE <= INT_MAX && BS_SIZE > 1024){
		int	errorCode = MPI_Recv(bs.data(), bs.size(), MPI_BYTE, source, tag, comm, &status);
		if( errorCode != MPI_SUCCESS /*errorCode != 0 */){
			printf("GC: MPI_Recv_Matrix: R%d received %d chars, from R%d, err = %d\n",
			(int)source, (int)bs.size(), (int)status.MPI_SOURCE, (int)status.MPI_ERROR);
		}
	}

	else{
		printf("Recv (4m)\n");
		
		int blocksCount = 4;
		int blocksLength[4] = {1, 1, 1, 1};
		MPI_Aint offsets[4];
		MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
		MPI_Datatype type_dsi4m;
		offsets[0] = offsetof(dsi4m, sizem1);
		offsets[1] = offsetof(dsi4m, sizem2);
		offsets[2] = offsetof(dsi4m, sizem3);
		offsets[3] = offsetof(dsi4m, sizem4);

		MPI_Type_create_struct(blocksCount, blocksLength, offsets, types, &type_dsi4m);
		MPI_Type_commit(&type_dsi4m);
		dsi4m recv;
		int	errorCode = MPI_Recv(&recv, 1, type_dsi4m, source /*rank*/, 0 /*tag*/, comm, &status);
		size_t size_bs = (size_t)recv.sizem1+(size_t)recv.sizem2+(size_t)recv.sizem3+(size_t)recv.sizem4;
		uchar *data = (uchar *) malloc( size_bs );
		size_t idx1 = (size_t)1*(size_t)recv.sizem1;
		size_t idx2 = (size_t)2*(size_t)recv.sizem1;
		size_t idx3 = (size_t)3*(size_t)recv.sizem1;

		errorCode += MPI_Recv(&data[0], 	recv.sizem1, MPI_UNSIGNED_CHAR, source, 11 /*tag*/, comm , &status);
		errorCode += MPI_Recv(&data[idx1],	recv.sizem2, MPI_UNSIGNED_CHAR, source, 12 /*tag*/, comm , &status);
		errorCode += MPI_Recv(&data[idx2],	recv.sizem3, MPI_UNSIGNED_CHAR, source, 13 /*tag*/, comm , &status);
		errorCode += MPI_Recv(&data[idx3],	recv.sizem4, MPI_UNSIGNED_CHAR, source, 14 /*tag*/, comm , &status);

		if( errorCode != MPI_SUCCESS /*errorCode != 0 */){
			printf("GC: MPI_Recv_Matrix: R%d received %d chars, from R%d, err = %d\n",
			(int)source, (int)bs.size(), (int)status.MPI_SOURCE, (int)status.MPI_ERROR);
		}

		bs.copy_stream( data, size_bs);
		bs.set_pos( 0 );
	}
	TBSHMBuilder builder;
	TMatrix* A = builder.build( bs ); 
	return A;
}

void 
// MPI_Send_HVector(TVector* vec, int destination, int tag){
MPI_Send_HVector(MPI_Comm comm, TVector* vec, int destination){
	int tag = 0;
	TByteStream bs;
	size_t size_bs = vec->bs_size();
	bs.set_size( size_bs );
	vec->write( bs );
	MPI_Send(bs.data(), bs.size(), MPI_BYTE, destination, tag, comm);
} 

TVector* 
MPI_Recv_Vector(MPI_Comm comm, int source, int blockSize){
	int tag = 0;
	MPI_Status status;
	int BS_SIZE;
	MPI_Probe( source, tag , comm, &status);
	MPI_Get_count( &status, MPI_BYTE, &BS_SIZE );
	TByteStream bs(BS_SIZE);
	MPI_Recv(bs.data(), bs.size(), MPI_BYTE, source, tag, comm, &status);
	hlib_vector_t vec = hlib_vector_build( blockSize, NULL );
	TVector* vec2 = ptrcast( (TVector*)vec, TVector * );
	vec2->read(bs);
	return vec2;
}