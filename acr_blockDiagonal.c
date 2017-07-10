#include "acr_blockDiagonal.h"
#include <assert.h>

pcrDiag
new_crDiag(int q, int n){
	pcrDiag Diag;
	int i,j;
	q = q + 1;
	Diag = (pcrDiag) calloc(1,sizeof(crDiag));
	assert (Diag != NULL);
	Diag->numLevels = q;
	Diag->level = (pcrLevel*) calloc(q,sizeof(pcrLevel)); 
	assert (Diag->level != NULL);
	j = q-1;
	for (i = 0; i < q; i++) {
		Diag->level[i] = (pcrLevel) calloc(1,sizeof(crLevel));
		assert (Diag->level[i] != NULL);
		Diag->level[i]->numBlocks = (int) pow(2,j);
		j--;
	}
	for (i = 0; i < q; i++) {
			Diag->level[i]->block = (pcrDenBlock*) calloc( (Diag->level[i]->numBlocks), sizeof(pcrDenBlock));
			assert (Diag->level[i]->block != NULL);
	}
	for (i = 0; i < q; i++) {
		for (j = 0; j < (Diag->level[i]->numBlocks) ; j++) {
			Diag->level[i]->block[j] = (pcrDenBlock) calloc(1, sizeof(crDenBlock));
			assert (Diag->level[i]->block[j] != NULL); // sure malloc
			Diag->level[i]->block[j]->size = n;
		}
	}
	return Diag;
}

void 
print_crDiag(pcrDiag AUXD, char* NAME) {
	int i,j;
	printf("crDiag:%s, %d levels \n\n",NAME,AUXD->numLevels);
	for (i = 0; i < (AUXD->numLevels); i++) {
		for (j = 0; j < (AUXD->level[i]->numBlocks) ; j++) {
			printf("%s->level[%d]->block[%d] \n",NAME,i,j);
			print_Hmatrix_dense(AUXD->level[i]->block[j]->e," ");
		}
		printf("\n");
	}
	printf("\n");
}

void 
print_crDiag_level(pcrDiag AUXD, int level, char* NAME) {
	int i,j;
	printf("crDiag:%s, %d levels \n\n",NAME,AUXD->numLevels);
	for (i = level; i <= level; i++) {
		for (j = 0; j < (AUXD->level[i]->numBlocks) ; j++) {
			printf("%s->level[%d]->block[%d] \n",NAME,i,j);
			print_Hmatrix_dense(AUXD->level[i]->block[j]->e," ");
		}
		printf("\n");
	}
	printf("\n");
}

double 
memory_crDiag_level(pcrDiag AUXD) {
	int i,j;
	double Diag_mem = 0.0;
	for (i = 0; i < (AUXD->numLevels); i++) {
		for (j = 0; j < (AUXD->level[i]->numBlocks) ; j++) {			
			Diag_mem+= hlib_matrix_bytesize(AUXD->level[i]->block[j]->e, NULL);
		}
	}
	return Diag_mem;
}

void
del_supermatrix(hlib_matrix_t A){
	hlib_matrix_free( A, NULL );
}

////////////////////////////////////////////////////////////////////////////////
// CR block vector functions
pVec
new_Vec(int size){
	pVec vec;
	int i,info;
	vec = (pVec) malloc(sizeof(acrVec));
	vec->size = size;
	vec->e = hlib_vector_build (size, &info);
	for (i = 0; i < size; i++) {
		hlib_vector_entry_set(vec->e, i, 0.0, &info);
	}
	return vec;
}

void
scale_Vec(pVec vec, double alpha){
	int info;
	hlib_vector_scale (	vec->e, alpha, &info);
}

void
del_Vec(pVec vec){
	int info;
	hlib_vector_free( vec->e, &info );
	freemem( vec );
	vec = NULL;
}

void
copy_Vec(pVec Va, pVec Vb){ 
	int info;
	Va->e = hlib_vector_copy (Vb->e, &info);
}

void
add_Vec(pVec y, double a, pVec x){
	int info;
	hlib_vector_axpy  ( a, x->e, y->e, &info );
}

void 
print_Vec(pVec vec, char *NAME) {
	int i,size = vec->size;
	int precision = 4;
	int width = 4 + 4;
	printf("%s (%d) = [\n",NAME,size);
	for (i = 0; i < size; i++) {
		printf("%-*.*f",width, precision,hlib_vector_entry_get(vec->e,i,NULL));
	}
	printf("\n]';\n");
}

void 
print_Vec_Entries(pVec vec, int ent, char *NAME) {
	int i,size = vec->size;
	int precision = 4;
	int width = 3 + 4;
	printf("%s (%d) = [\n",NAME,size);
	for (i = 0; i < ent; i++) {
		printf("%*.*f,",width, precision,hlib_vector_entry_get(vec->e,i,NULL));
	}
	printf("\n]\n");
}

void 
print_Vec_csv(pVec vec, char *NAME) {
	int i,size = vec->size;
	int precision = 4;
	int width = 3 + 4;
	printf("%s (%d) = [\n",NAME,size);
	for (i = 0; i < size; i++) {
		printf("%*.*f,",width, precision,hlib_vector_entry_get(vec->e,i,NULL));
	}
	printf("\n]\n");
}

////////////////////////////////////////////////////////////////////////////////
// CR block Diagonal Vector functions

pcrVec
new_crVec(int q, int n){
	int i,j;
	pcrVec crV;
	q = q + 1;
	crV = (pcrVec) malloc(sizeof(crVec));
	crV->numLevels = q;
	crV->level = (pcrVecLevel*) malloc(q*sizeof(pcrVecLevel)); 
	j = q-1;
	for (i = 0; i < q; i++) {
		crV->level[i] = (pcrVecLevel) malloc(sizeof(crVecLevel));
		crV->level[i]->numBlocks = (int) pow(2,j);
		j--;
	}
	for (i = 0; i < q; i++) {
		crV->level[i]->block = (pVec*) calloc( (crV->level[i]->numBlocks), sizeof(pVec));
	}
	for (i = 0; i < q; i++) {
		for (j = 0; j < (crV->level[i]->numBlocks) ; j++) {
			crV->level[i]->block[j] = new_Vec(n);
			crV->level[i]->block[j]->size = n;
		}
	}
	return crV;
}

void
del_crVec(pcrVec crV){
	int q,i,j;
	q = crV->numLevels;
	for (i = 0; i < q; i++) {
		for (j = 0; j < (crV->level[i]->numBlocks) ; j++) {
			del_Vec( crV->level[i]->block[j] );
		}
	}
	for (i = 0; i < q; i++) {
		freemem( crV->level[i]->block );
	}
	for (i = 0; i < q; i++) {
		freemem( crV->level[i] );
	}
	freemem( crV->level );
	crV->level = NULL;
	freemem( crV );
	crV = NULL;
}

void 
print_crVec(pcrVec crvec, char* varName){
	int i,j,k;	
	int precision = 6;
	int width = 3 + 6;
	printf("%s = [\n",varName);
	for (i = 0; i < crvec->numLevels; i++) {
		for (j = 0; j < (crvec->level[i]->numBlocks) ; j++) {
			for (k = 0; k < crvec->level[i]->block[j]->size; k++)	{
				printf(" %*.*f",width, precision, hlib_vector_entry_get(crvec->level[i]->block[j]->e,k,NULL) );
			}
		printf("|");
		}
	printf("\n"); 
	}
	printf( "]\n\n");
}

void
print_crVec_level(pcrVec crvec, int lev, char* varName){
	printf("%s = [\n",varName);
	int j,k,i = lev;
	int precision = 4;
	int width = 4 + 4;
	for (j = 0; j < (crvec->level[lev]->numBlocks) ; j++) {
		for (k = 0; k < crvec->level[i]->block[j]->size; k++)	{
			printf("%-*.*f",width, precision,hlib_vector_entry_get(crvec->level[i]->block[j]->e,k,NULL));
		}
		printf("|%d\n",j);
	}	
	printf("]';\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// CR vectors

pcrVecLevel
new_crVecLevel(int q, int iiq, int n){
	pcrVecLevel crVecLev;
	int i;
	crVecLev = (pcrVecLevel) malloc(sizeof(crVecLevel)); 
	crVecLev->numBlocks = (int) pow(2,q-iiq-1);
	crVecLev->block = (pVec*) calloc( (crVecLev->numBlocks), sizeof(pVec));
	for (i = 0; i < (crVecLev->numBlocks) ; i++) {
		crVecLev->block[i] = new_Vec(n);
		crVecLev->block[i]->size = n;
	}
	return crVecLev;
}

void
del_crVecLevel(pcrVecLevel crVecLev){
	int i;
	for (i = 0; i < (crVecLev->numBlocks) ; i++) {
		del_Vec( crVecLev->block[i] );
	}
}

void 
print_crVecLevel(pcrVecLevel crVecLev, char* varName){
	int j,k;
	int precision = 4;
	int width = 3 + 4;
	printf("%s = [\n",varName);
	for (j = 0; j < (crVecLev->numBlocks) ; j++) {
		for (k = 0; k < crVecLev->block[j]->size; k++)	{
			printf(" %*.*f",width, precision,hlib_vector_entry_get(crVecLev->block[j]->e,k,NULL));

		}
	printf("|");
	}
	printf( "\n]\n\n");
}

void 
copy_crVecLevel(pcrVecLevel VecLevA, pcrVecLevel VecLevB ){
	int i;
	for (i = 0; i < (VecLevB->numBlocks); i++) {
		copy_Vec(VecLevA->block[i], VecLevB->block[i]);
	}
}

void
dofreemem(void *p, const char *file, int line) {
  if(p == NULL) {
    (void) fprintf(stderr,
       "Trying to free NULL pointer.\n"
       "File %s, line %d\n",
       file, line);
    abort();
  }
  free(p);
}