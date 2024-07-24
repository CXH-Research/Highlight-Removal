# include<RPCA.h>
void PCP(Eigen::MatrixXd &AY, Eigen::MatrixXd &L, Eigen::MatrixXd &S, double mu, double lambda)
{
    int nr = AY.rows();
    int nc = AY.cols();

    if (mu == -1)
        mu = nr * nc*1.0f / (4 * AY.lpNorm<1>());
    if (lambda == -1)
        lambda = 1.0f / sqrt((std::max(nr, nc)));

    if (S.data() == NULL)
        S = Eigen::MatrixXd::Zero(nr, nc);

    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(nr, nc);
    double sca = 1.0f / mu;
    Record record;
    record.iteration = 0;
    record.oldRank = 0;
    record.Vold = Eigen::MatrixXd();
    record.SVDnPower = 1;
    record.SVDoffset = 5;

    bool BREAK = false;
    int maxIts = 1000;
    double nrmLS, perc;
    Eigen::MatrixXd dL, dS;
    double tol = 1e-5;
    for (int i = 0; i < maxIts; i++)
    {
        nrmLS = CalcFNorm(L, S);

        //解L最小化的问题
        dL = L;
        L = AY - S + sca * Y;
        projNuclear(L, 1.0f/mu, record);
        dL = L - dL;

        //解S最小化的问题
        dS = S;
        S = AY - L + sca * Y;
        Shrinkage(S, lambda/mu);
        dS = S - dS;

        perc = CalcFNorm(dL, dS) / (1 + nrmLS);
        if (perc < tol)
            break;
        Y = Y + mu * (AY - L - S);
    }
}
