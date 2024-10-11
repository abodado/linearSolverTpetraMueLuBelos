#include <chrono>
#include <sys/resource.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>

#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_ParameterListInterpreter.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosBlockGmresSolMgr.hpp>

using Teuchos::RCP;
using SC = double;
using LO = Tpetra::Map<>::local_ordinal_type;
using GO = Tpetra::Map<>::global_ordinal_type;
using NO = Tpetra::Map<>::node_type;
using map_type = Tpetra::Map<LO, GO, NO>;
using mtx_type = Tpetra::CrsMatrix<SC, LO, GO, NO>;
using multiVect_type = Tpetra::MultiVector<SC, LO, GO, NO>;
using op_type = Tpetra::Operator<SC, LO, GO, NO>;
using crsReader = Tpetra::MatrixMarket::Reader<mtx_type>;

std::string mtxFileName = "A.mtx";
std::string bFileName = "b.mtx";
std::string xFileName = "x.mtx";
std::string coordFileName = "coordinates.mtx"; // Coordinate file

int main(int argc, char *argv[])
{
  // getrusage variables
  struct rusage memUsage;
  int memStatus;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);

  {
    RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const size_t myRank = comm->getRank();

    // Read matrix A from Matrix Market file
    RCP<mtx_type> mA = crsReader::readSparseFile(mtxFileName, comm, false);
    RCP<const map_type> rowMap = mA->getRowMap();
   /*
    // Construct a Map that puts approximately the same number of
    // equations on each processor.
    const GO indexBase = 0;
    const GO numGlobalElements = 1000;
    RCP<const map_type> map = rcp (new map_type (numGlobalElements, indexBase, comm));

    //const size_t numMyElements = map->getLocalNumElements ();
    const size_t numMyElements = rowMap->getLocalNumElements ();
    std::cout  << "My Rank: " << myRank << std::endl;
    std::cout  << "number of elements on this process: " << numMyElements << std::endl;
  */
    /*
    // Create a map
    const GO numGlobalElements = 1000;
    RCP<const map_type> map = Tpetra::createUniformContigMapWithNode<LO, GO, NO>(numGlobalElements, comm);

    // Create a CrsMatrix
    RCP<mtx_type> A = Tpetra::createCrsMatrix<SC, LO, GO, NO>(map);

    // Fill the matrix (example: tridiagonal matrix)
    for (GO globalRow = map->getMinGlobalIndex(); globalRow <= map->getMaxGlobalIndex(); ++globalRow) {
      if (globalRow > 0) {
        A->insertGlobalValues(globalRow, Teuchos::tuple(globalRow - 1), Teuchos::tuple(-1.0));
      }
      A->insertGlobalValues(globalRow, Teuchos::tuple(globalRow), Teuchos::tuple(2.0));
      if (globalRow < numGlobalElements - 1) {
        A->insertGlobalValues(globalRow, Teuchos::tuple(globalRow + 1), Teuchos::tuple(-1.0));
      }
    }*/

    mA->fillComplete();

    // Read coordinates from Matrix Market file
    /*RCP<multiVect_type> coordinates;
    if (!coordFileName.empty()) {
      coordinates = Tpetra::MatrixMarket::Reader<multiVect_type>::readDenseFile(coordFileName, rowMap->getComm(), rowMap);
    }*/

    // Create MueLu preconditioner
    RCP<Teuchos::ParameterList> paramList = Teuchos::parameterList();

    paramList->set("verbosity", "high");
    paramList->set("max levels", 4);
    paramList->set("coarse: max size", 5000);
    paramList->set("multigrid algorithm", "sa");
    paramList->set("sa: damping factor", 1.33);
    paramList->set("reuse: type", "full");
    paramList->set("smoother: type", "RELAXATION");
    paramList->sublist("smoother: params").set("relaxation: type", "Gauss-Seidel");
    paramList->sublist("smoother: params").set("relaxation: sweeps", 3);
    paramList->sublist("smoother: params").set("relaxation: damping factor", 1.0);
    paramList->sublist("smoother: params").set("relaxation: zero starting solution", true);
    paramList->sublist("level 1").set("sa: damping factor", 0.0);
    paramList->set("aggregation: type", "uncoupled");
    paramList->set("aggregation: min agg size", 4);
    paramList->set("aggregation: max agg size", 36);
    paramList->set("aggregation: drop tol", 0.04);
    paramList->set("repartition: enable", false);
    paramList->set("repartition: partitioner", "zoltan2");
    paramList->set("repartition: start level", 1);
    paramList->set("repartition: min rows per proc", 20000000);
    paramList->set("repartition: target rows per proc", 0);
    paramList->set("repartition: max imbalance", 1.1);
    paramList->set("repartition: remap parts", true);
    paramList->set("repartition: rebalance P and R", false);
    paramList->sublist("repartition: params").set("algorithm", "multijagged");

    // Get memory status
    memStatus = getrusage(RUSAGE_SELF, &memUsage);
    long residentMem = memUsage.ru_maxrss;
    std::cout << "residentMem[MB]: " << residentMem/1024 << "; rank: " << myRank << std::endl;

    RCP<op_type> M = MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(mA, *paramList);

    // Get memory status
    memStatus = getrusage(RUSAGE_SELF, &memUsage);
    residentMem = memUsage.ru_maxrss;
    std::cout << "residentMem[MB]: " << residentMem/1024 << "; rank: " << myRank << std::endl;

    // Create multivector, X, for the linear system
    RCP<multiVect_type> vX = Tpetra::createMultiVector<SC>(rowMap, 1);
    // Read the X vector from Matrix Market file
    if (!xFileName.empty()) {
      vX = Tpetra::MatrixMarket::Reader<multiVect_type>::readDenseFile(xFileName, rowMap->getComm(), rowMap);
    } else {
      // Set initial X vector values to 293.15 Kelvin
      vX->putScalar(293.15);
    }

    // Create multivector, B, for the linear system
    RCP<multiVect_type> vB = Tpetra::createMultiVector<SC>(rowMap, 1);
    // Read the RHS vector B from Matrix Market file
    if (!bFileName.empty()) {
      vB = Tpetra::MatrixMarket::Reader<multiVect_type>::readDenseFile(bFileName, rowMap->getComm(), rowMap);
    } else {
      vB->putScalar(1.0);
    }

    // Set up the Belos linear problem
    Belos::LinearProblem<SC, multiVect_type, op_type> problem( mA , vX, vB);
    problem.setLeftPrec(M);
    problem.setProblem();

    // Create the Belos solver
    Teuchos::ParameterList belosList;
    belosList.set("Block Size", 1);
    belosList.set("Num Blocks", 40);
    belosList.set("Maximum Iterations", 50);
    belosList.set("Convergence Tolerance", 1e-8);
    belosList.set("Output Frequency", 1);
    belosList.set("Output Style", Belos::Brief);
    belosList.set("Verbosity", Belos::IterationDetails +
                  Belos::TimingDetails + Belos::FinalSummary +
                  Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);

    // Set up Belos solver using Teuchos parameter list
    Belos::BlockGmresSolMgr<SC, multiVect_type, op_type> solver(Teuchos::rcpFromRef(problem),
                                                                Teuchos::rcpFromRef(belosList));

    // Get memory status
    memStatus = getrusage(RUSAGE_SELF, &memUsage);
    residentMem = memUsage.ru_maxrss;
    std::cout << "residentMem[MB]: " << residentMem/1024 << "; rank: " << myRank << std::endl;

    // set up timer for solution
    auto time_start	= std::chrono::high_resolution_clock::now();
    // Solve the linear system
    Belos::ReturnType result = solver.solve();
    auto time_end	= std::chrono::high_resolution_clock::now();
    auto computeTime = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
    if (myRank == 0)
      std::cout << "Solver Took: " << computeTime/1e6 << " seconds; rank: " << myRank << std::endl;

    // Get memory status
    memStatus = getrusage(RUSAGE_SELF, &memUsage);
    residentMem = memUsage.ru_maxrss;
    std::cout << "residentMem[MB]: " << residentMem/1024 << "; rank: " << myRank << std::endl;

    // Write the solution vector to a Matrix Market file
    std::string mySolnVectfile = "SolnXVect.mtx";
    Tpetra::MatrixMarket::Writer<Tpetra::MultiVector<SC>>::writeDenseFile(mySolnVectfile, vX);

    if (result == Belos::Converged) {
      std::cout << "Converged!" << std::endl;
    } else {
      std::cout << "Not Converged!" << std::endl;
    }
  }

  return 0;
}
