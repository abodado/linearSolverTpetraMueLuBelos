#include <chrono>
#include <sys/resource.h>
//These headers are POSIX-specific, but very common
#include <sys/types.h>
#include <sys/stat.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

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
#include <BelosPseudoBlockGmresSolMgr.hpp>

#include <Galeri_XpetraProblemFactory.hpp>
#include <Galeri_XpetraMaps.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

using Teuchos::RCP;
using SC = double;
using LO = Tpetra::Map<>::local_ordinal_type;
using GO = Tpetra::Map<>::global_ordinal_type;
using NO = Tpetra::Map<>::node_type;
using map_type = Tpetra::Map<LO, GO, NO>;
using mtx_type = Tpetra::CrsMatrix<SC, LO, GO, NO>;
using multiVect_type = Tpetra::MultiVector<SC, LO, GO, NO>;
using xMV_Type = Xpetra::MultiVector<SC,LO,GO,NO>;
using op_type = Tpetra::Operator<SC, LO, GO, NO>;
using crsReader = Tpetra::MatrixMarket::Reader<mtx_type>;

std::string mtxFileName = "A.mtx";
std::string bFileName = "b.mtx";
std::string xFileName = "x.mtx";
std::string mueLuParamFile = "muelu_params.xml";

// Note: The coordinate matrix file must be a matrix market file in
// in column-oriented order. List all values of X first, then all values
// of Y, then all values of Z.
// Example for 2x2 square grid with elem length 0.5 units:
/*
%%MatrixMarket matrix array real general
4 3    # [Number of Coordinates] [Dimension]
0.25   # x0
0.75   # x1
0.25   # x2
0.75   # x3
0.25   # y0
0.25   # y1
0.75   # y2
0.75   # y3
0.0    # z0
0.0    # z1
0.0    # z2
0.0    # z3
*/

std::string coordFileName = "coordinates.mtx";

int main(int argc, char *argv[])
{
  // Check if the correct number of arguments is provided
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <g|gl|r>" << std::endl;
    std::cerr << "g: Run with 'galeri' coordinates" << std::endl;
    std::cerr << "r: Run with 'real' coordinates (supply coordinates.mtx)" << std::endl;
    return 1;
  }

  // Check if all required files exist
  std::ifstream fileA(mtxFileName);
  if (!fileA) {
    std::cerr << "Error: The file '" << mtxFileName << "' does not exist or cannot be opened." << std::endl;
    return 1; // Return a non-zero value to indicate an error
  }
  fileA.close();

  std::ifstream fileX(xFileName);
  if (!fileX) {
    std::cerr << "Error: The file '" << xFileName << "' does not exist or cannot be opened." << std::endl;
    return 1;
  }
  fileX.close();

  std::ifstream fileB(bFileName);
  if (!fileB) {
    std::cerr << "Error: The file '" << bFileName << "' does not exist or cannot be opened." << std::endl;
    return 1;
  }
  fileB.close();

  // Get the argument as a string
  std::string argument = argv[1];
  bool useGaleri = false;
  bool useRealCoord = false;

  // getrusage variables
  struct rusage memUsage;
  int memStatus;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);

  {
    RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const size_t myRank = comm->getRank();
    int NumProc = comm->getSize();

    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& fancyout = *fancy;
    fancyout.setOutputToRootOnly(0);

    // Check the argument and set the internal variable
    if (argument == "g") {
      useGaleri = true;
    } else if (argument == "r") {
      useRealCoord = true;
    }
    else {
      std::cerr << "Invalid argument. Use 'g' for galeri or 'r' for real coordinates." << std::endl;
      return 1;
    }

    // Read matrix A from Matrix Market file
    fancyout << "\nReading Matrix Market file, "<< mtxFileName << " ..." << std::endl;
    RCP<mtx_type> mA = crsReader::readSparseFile(mtxFileName, comm, false);
    RCP<const map_type> rowMap = mA->getRowMap();

    mA->fillComplete();

    // Create a Tpetra::MultiVector for coordinates
    RCP<multiVect_type> zoltanCoords = Teuchos::null;

    if (useRealCoord) {
      // Read/Load coordinates from Matrix Market
      std::ifstream fileC(coordFileName);
      if (!fileC) {
        std::cerr << "Error: The file '" << coordFileName << "' does not exist or cannot be opened." << std::endl;
        return 1;
      }
      fileC.close();
      fancyout << "Running with real (user-defined) coordinates..." << std::endl;
      fancyout << "Reading node coordinates from file..." << std::endl;
      zoltanCoords = Tpetra::MatrixMarket::Reader<multiVect_type>::readDenseFile(
            coordFileName,rowMap->getComm(),rowMap);
    }
    else if (useGaleri) {
      fancyout  << "Running with Galeri Coordinates..." << std::endl;
      int numGlobalEntries = rowMap->getGlobalNumElements();
      Teuchos::ParameterList GaleriList;
      GaleriList.set("nx", numGlobalEntries);
      GaleriList.set("ny", 1);
      GaleriList.set("mx", NumProc);
      GaleriList.set("my", 1);
      GaleriList.set("lx", 1.0); // length of x-axis
      GaleriList.set("ly", 1.0); // length of y-axis

      RCP<const Tpetra::Map<>> xpMap = Teuchos::null;
      RCP<Tpetra::MultiVector<>> xpCoord = Teuchos::null;
      // create map
      xpMap = Galeri::Xpetra::CreateMap<LO,GO,map_type>("Cartesian2D", comm, GaleriList);

      // create coordinates
      zoltanCoords = Galeri::Xpetra::Utils::CreateCartesianCoordinates
          <SC,LO,GO,map_type,Tpetra::MultiVector<>>("2D", xpMap, GaleriList);
    }
    else{
      std::cerr << "Coordinate info could not be determined. Program will exit." << std::endl;
      return 1;
    }

    Teuchos::ParameterList paramList;

    // Read MueLu param file if it exists
    // Check if the file exists using stat()
    struct stat buffer02;
    if (stat(mueLuParamFile.c_str(), &buffer02) == 0)  {
      // Load the parameter list from the XML file
      fancyout << "Reading MueLu parameters from xml file..." << std::endl;
      // Use Teuchos::getParametersFromXmlFile to read the parameters from the XML file
      RCP<Teuchos::ParameterList> xmlParams = Teuchos::getParametersFromXmlFile(mueLuParamFile);
      // Assign the parameters to your paramList
      paramList = *xmlParams;
    } else {
      // Create MueLu preconditioner if file not given
      paramList.set("verbosity", "high");
      paramList.set("max levels", 4);
      paramList.set("coarse: max size", 5000);
      paramList.set("multigrid algorithm", "sa");
      paramList.set("sa: damping factor", 1.33);
      paramList.set("reuse: type", "full");
      paramList.set("smoother: type", "RELAXATION");
      paramList.sublist("smoother: params").set("relaxation: type", "Gauss-Seidel");
      paramList.sublist("smoother: params").set("relaxation: sweeps", 3);
      paramList.sublist("smoother: params").set("relaxation: damping factor", 1.0);
      paramList.sublist("smoother: params").set("relaxation: zero starting solution", true);
      paramList.sublist("level 1").set("sa: damping factor", 0.0);
      paramList.set("aggregation: type", "uncoupled");
      paramList.set("aggregation: min agg size", 4);
      paramList.set("aggregation: max agg size", 36);
      paramList.set("aggregation: drop tol", 0.04);
      paramList.set("repartition: enable", true);
      paramList.set("repartition: partitioner", "zoltan2");
      paramList.set("repartition: start level", 1);
      paramList.set("repartition: min rows per proc", 50000);
      paramList.set("repartition: target rows per proc", 0);
      paramList.set("repartition: max imbalance", 1.1);
      paramList.set("repartition: remap parts", true);
      paramList.set("repartition: rebalance P and R", false);
      paramList.sublist("repartition: params").set("algorithm", "multijagged");
    }

    Teuchos::ParameterList& userData = paramList.sublist("user data");
    userData.set("Coordinates",zoltanCoords);

    // Get memory status
    memStatus = getrusage(RUSAGE_SELF, &memUsage);
    long residentMem = memUsage.ru_maxrss;
    fancyout << "residentMem[MB]: " << residentMem/1024 << "; rank: " << myRank << std::endl;

    RCP<op_type> M = MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(mA, paramList);

    // Get memory status
    memStatus = getrusage(RUSAGE_SELF, &memUsage);
    residentMem = memUsage.ru_maxrss;
    fancyout << "residentMem[MB]: " << residentMem/1024 << "; rank: " << myRank << std::endl;

    // Create multivector, X, for the linear system
    RCP<multiVect_type> vX = Tpetra::createMultiVector<SC>(rowMap, 1);
    // Read the X vector from Matrix Market file
    if (!xFileName.empty()) {
      vX = Tpetra::MatrixMarket::Reader<multiVect_type>::readDenseFile(xFileName, rowMap->getComm(), rowMap);
    } else {
      vX->putScalar(1.0);
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
    Belos::LinearProblem<SC, multiVect_type, op_type> problem(mA , vX, vB);
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

    belosList.set("Explicit Residual Scaling", "Norm of Preconditioned Initial Residual");

    // Set up Belos solver using Teuchos parameter list
    /*Belos::BlockGmresSolMgr<SC, multiVect_type, op_type> solver(Teuchos::rcpFromRef(problem),
                                                                Teuchos::rcpFromRef(belosList));*/

    Belos::PseudoBlockGmresSolMgr<SC,multiVect_type,op_type> solver(Teuchos::rcpFromRef(problem),
                                                                    Teuchos::rcpFromRef(belosList));

    // Get memory status
    memStatus = getrusage(RUSAGE_SELF, &memUsage);
    residentMem = memUsage.ru_maxrss;
    fancyout << "residentMem[MB]: " << residentMem/1024 << "; rank: " << myRank << std::endl;

    // set up timer for solution
    auto time_start	= std::chrono::high_resolution_clock::now();
    // Solve the linear system
    Belos::ReturnType result = solver.solve();
    auto time_end	= std::chrono::high_resolution_clock::now();
    auto computeTime = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
    fancyout << "Solver Took: " << computeTime/1e6 << " seconds" << std::endl;

    // Get memory status
    memStatus = getrusage(RUSAGE_SELF, &memUsage);
    residentMem = memUsage.ru_maxrss;
    fancyout << "residentMem[MB]: " << residentMem/1024 << "; rank: " << myRank << std::endl;

    // Write the solution vector to a Matrix Market file
    std::string mySolnVectfile = "SolnXVect.mtx";
    Tpetra::MatrixMarket::Writer<Tpetra::MultiVector<SC>>::writeDenseFile(mySolnVectfile, vX);

    if (result == Belos::Converged) {
      fancyout << "Solution converged!" << std::endl;
    } else {
      fancyout << "Solution did not converge!" << std::endl;
    }
  }

  return 0;
}
