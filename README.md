# linearSolverTpetraMueLuBelos
Solve linear system using Trilinos Tpetra, MueLu, and Belos

The initial version of this code uses Trilinos libraries and functions to read a system matrix A.mtx and RHS b.mtx (in MatrixMarket format) supplied by the user and then solve the system using Belos with MueLu as the preconditioner. Eventually, I would like to add functionality to read in a xml file of preconditioner settings to investigate the optimal settings for my linear system. I have a separate repository that uses Epetra, MueLu, and AztecOO so that one can compare the performance of the solvers.

The code uses Trilinos v16.0.0 (downloaded from the Trilinos repository). I have included the Trilinos build script as buildTrilinos.sh
