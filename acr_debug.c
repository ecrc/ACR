#include "acr_debug.h"
#include "io/TMatrixVis.hh"

int precision = 6;
int width = 4 + 6;

double get_Matrix_Size_MegaBytes(hlib_matrix_t A){
	double bytes = (double) hlib_matrix_bytesize(A, NULL);
	const double divider = 1024.0L;
	double MegaBytes = bytes/(divider*divider);
	return MegaBytes;
}

size_t get_Matrix_Memory(char const *NAME, hlib_matrix_t A) {
	size_t bytes = hlib_matrix_bytesize(A, NULL);
	const double divider = 1024.0L;

	if (bytes < divider)
		printf("%s = %1.2f B\n",NAME, (double)bytes);

	else if (bytes < (divider*divider))
		printf("%s = %1.2f KB\n",NAME, (double) (bytes/divider) );

	else if (bytes < (divider*divider*divider))
		printf("%s = %1.2f MB\n",NAME, (double)(bytes/(divider*divider)) );

	else if (bytes < (divider*divider*divider*divider))
		printf("%s = %1.2f GB\n",NAME, (double)(bytes/(divider*divider*divider)) );

	else if (bytes < (divider*divider*divider*divider*divider))
		printf("%s = %1.2f TB\n",NAME, (double)(bytes/(divider*divider*divider*divider)) );
	
	return bytes;
}

size_t get_Vector_Memory(char const *NAME, hlib_vector_t vec) {
	size_t bytes = hlib_vector_bytesize(vec, NULL);
	const double divider = 1024.0L;

	if (bytes < divider)
		printf("%s = %1.2f B\n",NAME, (double)bytes);

	else if (bytes < (divider*divider))
		printf("%s = %1.2f KB\n",NAME, (double) (bytes/divider) );

	else if (bytes < (divider*divider*divider))
		printf("%s = %1.2f MB\n",NAME, (double)(bytes/(divider*divider)) );

	else if (bytes < (divider*divider*divider*divider))
		printf("%s = %1.2f GB\n",NAME, (double)(bytes/(divider*divider*divider)) );

	else if (bytes < (divider*divider*divider*divider*divider))
		printf("%s = %1.2f TB\n",NAME, (double)(bytes/(divider*divider*divider*divider)) );
	return bytes;
}

void get_Bytes_Memory(char const *NAME, size_t bytes) {
	const double divider = 1024.0L;
	if (bytes < divider)
		printf("%s = %1.2f B\n",NAME, (double)bytes);

	else if (bytes < (divider*divider))
		printf("%s = %1.2f KB\n",NAME, (double) (bytes/divider) );

	else if (bytes < (divider*divider*divider))
		printf("%s = %1.2f MB\n",NAME, (double)(bytes/(divider*divider)) );

	else if (bytes < (divider*divider*divider*divider))
		printf("%s = %1.2f GB\n",NAME, (double)(bytes/(divider*divider*divider)) );

	else if (bytes < (divider*divider*divider*divider*divider))
		printf("%s = %1.2f TB\n",NAME, (double)(bytes/(divider*divider*divider*divider)) );
}

// Debug functions
void 
print_Hmatrix_dense_short(hlib_matrix_t A, char const *NAME) {
	int i,j,rows,cols;
	rows = 18;
	cols = 18;
	double entry;
	printf("%s (short: %d,%d)= [\n",NAME,rows,cols);
	for (i=0; i<rows; i++){
	 for(j=0; j<cols; j++){
	     if ( (i==rows-1) && (j==cols-1) ) {
			entry = hlib_matrix_entry_get(A, i, j, NULL);
			if (entry==0.0)
			printf("%*.*f", width, 0 , entry );
			else
			printf("%*.*f", width, precision , entry );
			printf("\n];");
		}
		else {
			entry = hlib_matrix_entry_get(A, i, j, NULL);
			if (entry==0.0)
				printf("%*.*f,", width, 0 , entry );
			else
				printf("%*.*f,", width, precision , entry );
		}
	 }
	printf( "\n");
	}
}

void 
print_Hmatrix_dense(hlib_matrix_t A, char const *NAME) {
	int i,j,rows,cols;
	double entry;
	rows = (int) hlib_matrix_rows (A, NULL);
	cols = (int) hlib_matrix_cols (A, NULL);

	printf("%s (%d,%d)= [\n",NAME,rows,cols);
	for (i=0; i<rows; i++) {
	 for(j=0; j<cols; j++) {
	     if ( (i==rows-1) && (j==cols-1) ) {
			entry = hlib_matrix_entry_get(A, i, j, NULL);
			if (entry==0.0){
				printf("%*.*f", width, precision, entry );
			}
			else
			printf("%*.*f", width, precision , entry );
			printf("\n];");
	     }
	    else {
			entry = hlib_matrix_entry_get(A, i, j, NULL);
			if (entry==0.0){
				printf("%*.*f,", width, precision, entry );
			}
			else
				printf("%*.*f,", width, precision , entry );
	    }
	 }
	    printf( "\n");
	}
}

void 
print_Vector(hlib_vector_t vec, char const *NAME){
	int i;
	int size = (int) hlib_vector_size(vec,NULL);
	printf("%s (%d) = [\n",NAME,size);
	for (i = 0; i < size; i++) {
		printf("%*.*f",width, precision,hlib_vector_entry_get(vec,i,NULL));
	}
	printf("\n]';\n");
}

void
visualize_H_Matrix(char const *NAME, hlib_matrix_t H_Mat, int vis_code){
	char NAME_AND_COMMAND[255];
	char NAME_AND_EXTENSION[124];
	sprintf(NAME_AND_EXTENSION,"%s.eps",NAME);

	// Print matrix size
	if (vis_code == 0){
		printf("%s = [%d x %d]\n", NAME, (int)hlib_matrix_rows(H_Mat,NULL), (int)hlib_matrix_cols(H_Mat,NULL));
	}

	// Print svd rank in each block
	else if (vis_code == 1){
		printf("%s = [%d x %d]\n", NAME, (int)hlib_matrix_rows(H_Mat,NULL), (int)hlib_matrix_cols(H_Mat,NULL));
		hlib_matrix_print_ps(H_Mat, NAME_AND_EXTENSION, HLIB_MATIO_SVD, NULL);
		system(NAME_AND_COMMAND);
	}
	
	// Print sparsity pattern (non-zero entries)
	else if (vis_code == 2){
		hlib_matrix_print_ps(H_Mat,NAME_AND_EXTENSION, HLIB_MATIO_PATTERN, NULL);
		system(NAME_AND_COMMAND);
		printf("%s = [%d x %d]\n",NAME, (int)hlib_matrix_rows(H_Mat,NULL), (int)hlib_matrix_cols(H_Mat,NULL));
	}

	// Print each entry of matrix
	else if (vis_code == 3){
		hlib_matrix_print_ps(H_Mat,NAME_AND_EXTENSION, HLIB_MATIO_ENTRY, NULL);
		system(NAME_AND_COMMAND);
		printf("%s = [%d x %d]\n",NAME, (int)hlib_matrix_rows(H_Mat,NULL), (int)hlib_matrix_cols(H_Mat,NULL));
	}
}

void 
checkSymmetry(hlib_matrix_t A, char const *NAME){
	if ( hlib_matrix_is_sym(A,NULL) == 1 ){
		printf("%s is Symmetric\n",NAME);
	}
	else{
		printf("%s is NOT Symmetric!\n",NAME);
	}
}