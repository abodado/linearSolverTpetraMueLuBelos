# linearSolverTpetraMueLuBelos
Solve linear system using Trilinos Tpetra, MueLu, and Belos

The initial version of this code uses Trilinos libraries and functions to read a system matrix A.mtx and RHS b.mtx (in MatrixMarket format) supplied by the user and then solve the system using Belos with MueLu as the preconditioner. Eventually, I would like to add functionality to read in a xml file of preconditioner settings to investigate the optimal settings for my linear system. I have a separate repository that uses Epetra, MueLu, and AztecOO so that one can compare the performance of the solvers.

The code uses Trilinos v16.0.0 (downloaded from the Trilinos repository). I have included the Trilinos build script as buildTrilinos.sh

The order of libraries to link can be tricky. First I have some system libraries, followed by the Trilinos libraries, and then lapack, blas, and mpi.

LIBS =  -L../../usr/lib64 \
        -L/usr/lib64  \
        -L$${MPIPATH}/lib \
		-lGL -lGLU -lqsa -lmpi -lgfortran \
        -L$${TRILINOSDIR}/lib64 -lmuelu -lmuelu-adapters \
		-lbelosxpetra -lbelostpetra -lbelos -lpamgen_extras \
		-lpamgen -lgaleri-xpetra \
		-lintrepid2 -lshards -lsacado -lteko -lanasazitpetra \
		-lanasazi -lstratimikos -lstratimikosbelos \
		-lstratimikosamesos2 \
		-lifpack2-adapters -lifpack2 \
		-lbelosxpetra -lbelostpetra -lbelos \
		-lamesos2 -ltacho -lzoltan2 \
		-ltrilinosss -lxpetra-sup -lxpetra \
		-lthyratpetra -lthyracore -lrtop \
		-ltpetraext -ltpetrainout -ltpetra -lkokkostsqr \
		-ltpetraclassic \
		-lzoltan -lkokkoskernels -lkokkosalgorithms \
		-lkokkoscore -lteuchoskokkoscomm -lteuchoskokkoscompat \
		-lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist \
		-lteuchosparser -lteuchoscore -lkokkoscontainers -lkokkoscore -llapack \
		-lblas -lmpi	
