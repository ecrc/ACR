#!/bin/bash

# Running ACR for a problem of N=32^3. H matrices are of size 32^2
# H matrices are built and operated to accuracy $HEPS
# Size of leaf nodes is $NMIN and admissibility condition parameter is $ETA
# Linear system data is at ../inputData/

SHARED_MEM_CORES=2
HDIM=32
CRDIM=32
NMIN=32
ETA=16
HEPS=1e-3

mpirun -np $CRDIM ../directACR.out $SHARED_MEM_CORES $HDIM $CRDIM $NMIN $ETA $HEPS