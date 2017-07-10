# Set the environment variable HPRO_PATH

INCLUDE_HPRO = -I$(HPRO_PATH)/include
INCLUDE_BOOST = -I$(HPRO_PATH)/aux/include/boost/graph -I$(HPRO_PATH)/aux/include/ -I$(HPRO_PATH)/aux/include/boost

LIB_HPRO = -L$(HPRO_PATH)/lib
LIB_METIS = -L$(HPRO_PATH)/aux/lib/
LIB_AUX = -Wl,-rpath=$(HPRO_PATH)/lib -Wl,-rpath=$(HPRO_PATH)/aux/lib

CC = mpiicpc
CFLAGS = -std=c++11 -O2 -debug none
INCLUDES = ${INCLUDE_HPRO} ${INCLUDE_BOOST} ${LIB_HPRO} ${LIB_METIS} ${LIB_AUX}
LIBS_FLAGS = -lhpro -ltbb -lmetis -lscotch -lscotcherr\
-lboost_filesystem -lboost_iostreams -lboost_system -mkl=sequential

OBJECTS = \
	acr_debug.c \
	acr_blockDiagonal.c \
	acr_Hwrappers.c \
	acr_mpi.c \
	acr_setup.c \
	acr_solver.c

all:
	$(CC) directACR.c $(OBJECTS) $(INCLUDES) $(CFLAGS) -o directACR.out $(LIBS_FLAGS)

clean:
	rm -f *.eps *.ps *.out