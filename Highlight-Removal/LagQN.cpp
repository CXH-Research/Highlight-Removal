#include<RPCA.h>
void LagQN(Eigen::MatrixXd &AY, Eigen::MatrixXd &L, Eigen::MatrixXd &S, double lambda_S)
{
    double tol = 1e-4, tau = -1;
    int n1 = AY.rows();
    int n2 = AY.cols();
    //L.setZero();
    //S.setZero();
    bool SMALL = (n1*n2 < pow(50, 2));
    bool MEDIUM = (n1*n2 <= pow(200, 2) && !SMALL);
    bool LARGE = (n1*n2 <= pow(1000, 2) && !SMALL && !MEDIUM);
    bool HUGE_Matrix = (n1*n2 > std::pow(1000, 2));

    int maxIts = 1e3*(SMALL | MEDIUM) + 400 * LARGE + 200 * HUGE_Matrix;
    int printEvery = 100 * SMALL + 50 * MEDIUM + 5 * LARGE + 1 * HUGE_Matrix;

    double Lip;
    double stepsizeQN;

    int restart, trueObj;
    bool displayTime, SVDwarmStart;
    double lambda, lambdaL, lambdaS, stepsize;

    Lip = 2;
    stepsizeQN = 0.8 * 2 / Lip;
    restart = INT_MIN;
    trueObj = 0;
    displayTime = LARGE | HUGE_Matrix;
    SVDwarmStart = true;
    lambda = abs(tau);

    lambdaL = lambda;
    lambdaS = lambda * lambda_S;

    Record record;
    record.iteration = 0;
    record.oldRank = 0;
    record.Vold = Eigen::MatrixXd();
    record.SVDnPower = 1;
    record.SVDoffset = 5;

    Eigen::MatrixXd errHist, L_old, S_old, dL, dS, Grad, S_temp;

    stepsize = 1 / Lip;
    errHist = Eigen::MatrixXd::Zero(maxIts, 2);
    Grad = Eigen::MatrixXd::Zero(n1, n2);
    dL = Eigen::MatrixXd::Zero(n1, n2);
    dS = Eigen::MatrixXd::Zero(n1, n2);
    L_old = L;
    S_old = S;

    bool BREAK = false;
    int k, rnk;
    Grad = L + S - AY;

    Eigen::MatrixXd temp;

    cv::Mat SShow, LShow;
    cv::Mat errHistShow;
    //project = @(L, S, varargin) projectMax(lambda, lambda*lambda_S, SVDopts, L, S, varargin{ : });
    double tauS = abs(lambdaS*stepsizeQN);
    double tauL = -abs(lambdaL*stepsizeQN);
    for (k = 0; k < maxIts; k++)
    {
        dL = L - L_old;
        S_old = S;


        S_temp = S - stepsizeQN * (Grad + dL);
        Shrinkage(S_temp, tauS);
//        cv::eigen2cv(S_temp, SShow);
        dS = S_temp - S_old;


        L_old = L;
        L = L - stepsizeQN * (Grad + dS);
        rnk = projNuclear(L, tauL, record);
//        cv::eigen2cv(L, LShow);

        dL = L - L_old;
        S = S - stepsizeQN * (Grad + dL);
        Shrinkage(S, tauS);
//        cv::eigen2cv(S, SShow);

        Grad = L + S - AY;

        double res = Grad.norm();
        std::cout<<res<<std::endl;
        errHist(k, 0) = res;
        errHist(k, 1) = 1.0 / 2 * (res*res);
        cv::eigen2cv(errHist, errHistShow);
        if (k > 0 && abs(errHist(k, 0) - errHist(k - 1, 0)) / res < tol)
            BREAK = true;
        if (BREAK)
            break;
    }
}
