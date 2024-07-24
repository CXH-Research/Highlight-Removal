#include<RPCA.h>
void FPCP(Eigen::MatrixXd &AY, Eigen::MatrixXd &Lowrank, Eigen::MatrixXd &Sparse, int loops)
{
    int nr = AY.rows();
    int nc = AY.cols();
    double lambda = 1.0f / sqrt(MAX(nr, nc));
    double lambdaFactor = 1;
    double rankThreshold = 0.01;
    int rank0 = 1;
    int inc_rank = 1;

    int rank = rank0;
    std::vector<int> RankVec;
    RankVec.push_back(rank);
    Eigen::MatrixXd U, S, V;
    Eigen::MatrixXd opts;
    randomizedSVD(AY, U, S, V, rank, MIN(MIN(rank + SVDOFFSET, nr), nc), SVDNPOWER, opts);

    //U*S
    Eigen::MatrixXd TempMatrix = Eigen::MatrixXd::Zero(U.rows(), S.cols());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, U.rows(), S.cols(), U.cols(), 1, (double*)U.data(),
        U.rows(), (double*)S.data(), S.rows(), 1, (double*)TempMatrix.data(), U.rows());
    //L=U*S*VT
    Lowrank.setZero();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, TempMatrix.rows(), V.rows(), TempMatrix.cols(), 1, (double*)TempMatrix.data(),
        TempMatrix.rows(), (double*)V.data(), V.rows(), 1, (double*)Lowrank.data(), TempMatrix.rows());

    Sparse = AY - Lowrank;
    Shrinkage(Sparse, lambda);
    double rho;
    for (int k = 2; k <= loops; k++)
    {
        if (inc_rank == 1)
        {
            lambda = lambda * lambdaFactor;
            rank = rank + 1;
        }
        Lowrank = AY - Sparse;
        randomizedSVD(Lowrank, U, S, V, rank, ell(rank + SVDOFFSET, nr, nc), SVDNPOWER, opts);
        double LastVal = S(rank - 1, rank - 1), Sum = 0;
        for (int i = 0; i < rank - 1; i++)
            Sum += S(i, i);
        RankVec.push_back(rank);
        rho = LastVal / Sum;
        if (rho < rankThreshold)
            inc_rank = 0;
        else
            inc_rank = 1;
        TempMatrix = Eigen::MatrixXd::Zero(U.rows(), S.rows());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, U.rows(), S.cols(), U.cols(), 1, (double*)U.data(),
            U.rows(), (double*)S.data(), S.rows(), 1, (double*)TempMatrix.data(), U.rows());
        //L=U*S*VT
        Lowrank.setZero();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, TempMatrix.rows(), V.rows(), TempMatrix.cols(), 1, (double*)TempMatrix.data(),
            TempMatrix.rows(), (double*)V.data(), V.rows(), 1, (double*)Lowrank.data(), TempMatrix.rows());

        Sparse = AY - Lowrank;
        Shrinkage(Sparse, lambda);
    }
}
