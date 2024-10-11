#!/bin/bash

export CC=gcc
export OMPI_CC=gcc
export CXX=g++
export OMPI_CXX=g++
export FC=gfortran
export OMPI_FC=gfortran

export TRILINOS_ROOT=/s/bsweetma/trilinos
export TRILINOS_HOME=${TRILINOS_ROOT}/Trilinos-trilinos-release-16-0-0
export TrilinosBuildDir=${TRILINOS_HOME}/buildTrilinos
export TrilinosInstallDir=${TRILINOS_HOME}/__INSTALL
mkdir ${TrilinosBuildDir}
mkdir ${TrilinosInstallDir}

# Get this script's name so that we can copy it to the INSTALL folder
# Saving a copy of this config script reminds us how we configured this 
# particular build of Trilinos
script_name=$(basename "$0")
cp "$0" "$TrilinosInstallDir/$script_name"
# Optional: Print message to user that config script was copied.
echo "Script '$script_name' copied to '$TrilinosInstallDir'"
cd ${TrilinosBuildDir}

export MPI_ROOT=/apps/mpi/openmpi/openmpi-4.0.2
export BLASPATH=${TRILINOS_ROOT}/lapack-3.10.1/INSTALL/lib64

# Don't include!
#-D CMAKE_CXX_FLAGS:STRING="-fopenmp" \
#-D CMAKE_C_FLAGS:STRING="-fopenmp" \
#-D TPL_BLAS_LIBRARIES="-L${BLASPATH} -lblas -L/usr/lib64/;-lgfortran;-lgomp;-lm" \
#-D CMAKE_EXE_LINKER_FLAGS:STRING="-lgfortran -lgomp -lm" \

cmake \
-D CMAKE_C_COMPILER=$(which mpicc) \
-D CMAKE_CXX_COMPILER=$(which mpicxx) \
-D TPL_ENABLE_MPI:BOOL=ON \
-D TPL_ENABLE_BLAS:BOOL=ON \
-D TPL_ENABLE_LAPACK:BOOL=ON \
-D MPI_BASE_DIR=${MPI_ROOT} \
-D BLAS_LIBRARY_DIRS=${BLASPATH} \
-D LAPACK_LIBRARY_DIRS=${BLASPATH} \
-D CMAKE_EXE_LINKER_FLAGS:STRING="-lgfortran -lm" \
-D Trilinos_ENABLE_OpenMP:BOOL=OFF \
-D Trilinos_ENABLE_Tpetra:BOOL=ON \
-D Tpetra_ENABLE_TESTS=ON \
-D Trilinos_ENABLE_Kokkos:BOOL=ON \
-D Trilinos_ENABLE_MueLu:BOOL=ON \
-D Trilinos_ENABLE_Zoltan2:BOOL=ON \
-D Trilinos_ENABLE_Amesos2:BOOL=ON \
-D Trilinos_ENABLE_Ifpack2:BOOL=ON \
-D Trilinos_ENABLE_Belos:BOOL=ON \
-D Kokkos_ENABLE_OPENMP:BOOL=OFF \
-D Kokkos_ARCH_NATIVE:BOOL=ON \
-D Trilinos_ENABLE_Xpetra:BOOL=ON \
-D Trilinos_ENABLE_Teuchos:BOOL=ON \
-D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
-D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=ON \
-D CMAKE_BUILD_TYPE:STRING="RELEASE" \
-D Trilinos_HIDE_DEPRECATED_CODE:BOOL=ON \
-D Trilinos_ENABLE_TESTS:BOOL=OFF \
-D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
-D CMAKE_INSTALL_PREFIX=${TrilinosInstallDir} \
${TRILINOS_HOME}

make -j16 install
