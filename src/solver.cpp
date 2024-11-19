#include <chrono>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <MatrixMarket_Tpetra.hpp>  // Updated to use MatrixMarket_Tpetra.hpp

#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Hierarchy.hpp>
#include <MueLu_Utilities.hpp>

#include <Xpetra_TpetraMultiVector.hpp>
#include <Xpetra_TpetraMap.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_CrsMatrix.hpp>

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

// Define a struct to hold convergence criteria
struct ConvergenceCriteria {
  int maxIterations;
  double tolerance;
};

// Function to perform iterations and check convergence
void iterateWithConvergenceCheck(RCP<MueLu::Hierarchy<SC, LO, GO, NO>> H,
                                 RCP<Xpetra::Matrix<SC, LO, GO, NO>> A,
                                 RCP<Xpetra::MultiVector<SC, LO, GO, NO>> B,
                                 RCP<Xpetra::MultiVector<SC, LO, GO, NO>> X,
                                 const ConvergenceCriteria& criteria) {
  for (int i = 0; i < criteria.maxIterations; ++i) {
    H->Iterate(*B, *X, 1);

    // Calculate residual R = AX - B
    RCP<Xpetra::MultiVector<SC, LO, GO, NO>> R = Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(B->getMap(), 1);
    A->apply(*X, *R);
    R->update(-1.0, *B, 1.0);

    // Compute the 2-norm of the residual
    Teuchos::Array<Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
    R->norm2(norms);
    std::cout << "V-Cycle " << i + 1 << " residual norm: " << norms[0] << std::endl;

    if (norms[0] < criteria.tolerance) {
      std::cout << "Converged after " << i + 1 << " V-cycles with residual norm: " << norms[0] << std::endl;
      break;
    }
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);

  {
    RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const size_t myRank = comm->getRank();

    // Read matrix A from Matrix Market file
    RCP<mtx_type> A = crsReader::readSparseFile(mtxFileName, comm, false);
    RCP<const map_type> rowMap = A->getRowMap();

    A->fillComplete();

    // Create multivectors for the linear system
    RCP<multiVect_type> X = Tpetra::createMultiVector<SC>(rowMap, 1);
    RCP<multiVect_type> B = Tpetra::createMultiVector<SC>(rowMap, 1);

    // Read the RHS vector B from Matrix Market file
    if (!bFileName.empty()) {
      B = Tpetra::MatrixMarket::Reader<multiVect_type>::readDenseFile(bFileName, rowMap->getComm(), rowMap);
    } else {
      B->putScalar(1.0);
    }

    // Create MueLu preconditioner (and solver)
    RCP<Teuchos::ParameterList> paramList = Teuchos::parameterList();
    paramList->set("verbosity", "high");
    paramList->set("max levels", 3);
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

    // MueLu as a preconditioner in Belos
    {
      RCP<op_type> M = MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(A, *paramList);

      // Set up the Belos linear problem
      Belos::LinearProblem<SC, multiVect_type, op_type> problem(A, X, B);
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
      belosList.set("Verbosity", Belos::IterationDetails + Belos::TimingDetails
                    + Belos::FinalSummary + Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);

      // Set up Belos solver using Teuchos parameter list
      Belos::BlockGmresSolMgr<SC, multiVect_type, op_type> solver(Teuchos::rcpFromRef(problem), Teuchos::rcpFromRef(belosList));

      // Solve the linear system
      Belos::ReturnType result = solver.solve();

      std::string mySolnVectfile = "SolnXVect_Belos.mtx";
      Tpetra::MatrixMarket::Writer<Tpetra::MultiVector<SC>>::writeDenseFile(mySolnVectfile, X);

      if ( myRank == 0 ){
        if (result == Belos::Converged) {
          std::cout << "Belos with MueLu preconditioner Converged!" << std::endl;
        } else {
          std::cout << "Belos with MueLu preconditioner Not Converged!" << std::endl;
        }
      }
    }

    // MueLu as a stand-alone solver with multiple V-cycles
    /*{
      // Convert Tpetra objects to Xpetra
      RCP<Xpetra::TpetraCrsMatrix<SC, LO, GO, NO>> xpetra_A_tpetra = Teuchos::rcp(new Xpetra::TpetraCrsMatrix<SC, LO, GO, NO>(A));
      RCP<Xpetra::CrsMatrix<SC, LO, GO, NO>> xpetra_A = Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrix<SC, LO, GO, NO>>(xpetra_A_tpetra);
      RCP<Xpetra::Matrix<SC, LO, GO, NO>> wrapped_A = Teuchos::rcp(new Xpetra::CrsMatrixWrap<SC, LO, GO, NO>(xpetra_A));

      RCP<Xpetra::TpetraMultiVector<SC, LO, GO, NO>> xpetra_X_tpetra = Teuchos::rcp(new Xpetra::TpetraMultiVector<SC, LO, GO, NO>(X));
      RCP<Xpetra::MultiVector<SC, LO, GO, NO>> xpetra_X = Teuchos::rcp_dynamic_cast<Xpetra::MultiVector<SC, LO, GO, NO>>(xpetra_X_tpetra);

      RCP<Xpetra::TpetraMultiVector<SC, LO, GO, NO>> xpetra_B_tpetra = Teuchos::rcp(new Xpetra::TpetraMultiVector<SC, LO, GO, NO>(B));
      RCP<Xpetra::MultiVector<SC, LO, GO, NO>> xpetra_B = Teuchos::rcp_dynamic_cast<Xpetra::MultiVector<SC, LO, GO, NO>>(xpetra_B_tpetra);

      // Ensure all RCPs are valid
      if (wrapped_A.is_null() || xpetra_X.is_null() || xpetra_B.is_null()) {
        std::cerr << "Error: Null RCP encountered during conversion to Xpetra" << std::endl;
        return EXIT_FAILURE;
      }

      // Create MueLu hierarchy
      RCP<MueLu::Hierarchy<SC, LO, GO, NO>> H = MueLu::CreateXpetraPreconditioner<SC, LO, GO, NO>(wrapped_A, *paramList);

      // Define convergence criteria
      ConvergenceCriteria criteria;
      criteria.maxIterations = 50;  // Maximum number of V-cycles
      criteria.tolerance = 1e-8;    // Convergence tolerance

      auto iterStart = std::chrono::high_resolution_clock::now();
      // Perform iterations with convergence check
      iterateWithConvergenceCheck(H, wrapped_A, xpetra_B, xpetra_X, criteria);
      auto iterEnd = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(iterEnd - iterStart).count();
      std::cout << "multigrid solver run time = " << duration << "; rank = " << myRank << std::endl;

      // Convert xpetra_X back to Tpetra::MultiVector for writing to file
      RCP<multiVect_type> tpetra_X = Teuchos::rcp_dynamic_cast<Xpetra::TpetraMultiVector<SC, LO, GO, NO>>(xpetra_X)->getTpetra_MultiVector();

      std::string mySolnVectfile = "SolnXVect_MueLu.mtx";
      Tpetra::MatrixMarket::Writer<Tpetra::MultiVector<SC>>::writeDenseFile(mySolnVectfile, tpetra_X);
    }*/
  }

  return 0;
}

