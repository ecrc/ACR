#ifndef _acr_blockDiagonal_h
#define _acr_blockDiagonal_h

#include "hlib-c.h"
#include "acr_debug.h"
#include <stdio.h>

// Matrix Block
typedef struct _crDenBlock crDenBlock;
typedef crDenBlock *pcrDenBlock;

struct _crDenBlock {
   int size;
   hlib_matrix_t e;
};

typedef struct _crLevel crLevel;
typedef crLevel *pcrLevel;
 
struct _crLevel {
   int numBlocks;
   pcrDenBlock* block;
};   

typedef struct _crDiag crDiag;
typedef crDiag *pcrDiag;
 
struct _crDiag {
   int numLevels;
   pcrLevel* level;
};

////////////////////////////////////////////////////////////////////////////////

// Simple Vector
typedef struct _Vec acrVec;
typedef acrVec *pVec;
 
struct _Vec {
   int size;
   hlib_vector_t e;
};

// Level Vector
typedef struct _crVecLevel crVecLevel;
typedef crVecLevel *pcrVecLevel;
 
struct _crVecLevel {
   int numBlocks;
   pVec* block;
};

typedef struct _crVec crVec;
typedef crVec *pcrVec;
 
struct _crVec {
   int numLevels;
   pcrVecLevel* level;
};


////////////////////////////////////////////////////////////////////////////////
// CR block Diagonal Matrix functions

pcrDiag
new_crDiag(int q, int n);

void 
print_crDiag(pcrDiag AUXD, char* NAME);

void 
print_crDiag_level(pcrDiag AUXD, int level, char* NAME);

double
memory_crDiag_level(pcrDiag AUXD);

void
del_crDiag(pcrDiag Diag);

void
del_supermatrix(hlib_matrix_t A);

////////////////////////////////////////////////////////////////////////////////
// CR block vector functions
pVec
new_Vec(int size);

void
scale_Vec(pVec vec, double alpha);

void
del_Vec(pVec vec);

void
copy_Vec(pVec Va, pVec Vb);

void
add_Vec(pVec y, double a, pVec x);

void 
print_Vec(pVec vec, char *NAME);

////////////////////////////////////////////////////////////////////////////////
// CR block Diagonal Vector functions

pcrVec
new_crVec(int q, int n);

void
del_crVec(pcrVec crV);

void 
print_crVec(pcrVec crvec, char* varName);

void
print_crVec_level(pcrVec crvec, int lev, char* varName);

////////////////////////////////////////////////////////////////////////////////
// CR vectors

pcrVecLevel
new_crVecLevel(int q, int iiq, int n);

void
del_crVecLevel(pcrVecLevel crVecLev);

void 
print_crVecLevel(pcrVecLevel crVecLev, char* varName);

void 
copy_crVecLevel(pcrVecLevel VecLevA, pcrVecLevel VecLevB);


////////////////////////////////////////////////////////////////////////////////
#ifdef freemem
#undef freemem
#endif
#define freemem(p) dofreemem(p,__FILE__,__LINE__),p=NULL
/* Do not use dofreemem directly! */
void
dofreemem(void *p, const char *file, int line);

#endif