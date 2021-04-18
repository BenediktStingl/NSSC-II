#include "header.hpp"

std::vector<double> jacobi(long Y, long X, double h, long iterations)
{
    std::vector<double> U   = generate_U(Y, X, h);
    std::vector<double> U_k = generate_U(Y, X, h);
    for (long k = 0; k < iterations; k++)
    {
        for (long i = 1; i < Y-1; i++)
        {
            for (long j = 1; j < X-1; j++)
            {
                long   I = X*i + j;
                double F = (4 * M_PI*M_PI * h*h);
                double RHS = F * sin( 2*M_PI*(j*h) ) * sinh( 2*M_PI*(i*h) );
                double sum = (-1)*U[I-1] + (-1)*U[I-X] + (-1)*U[I+1] + (-1)*U[I+X];
                U_k[I] = (RHS - sum) / (4+F);
            }
        }
        U = U_k;
    }
    return U;
}

int main(int argc, char *argv[])
{
    long resolution = read_parameter(1, argc, argv, 50);
    long iterations = read_parameter(2, argc, argv, 500);

    long Y = resolution;
    long X = 2*resolution-1;
    double h = 1.0/(resolution-1);

    auto start   = std::chrono::steady_clock::now();
    std::vector<double> U = jacobi(Y, X, h, iterations);
    auto end     = std::chrono::steady_clock::now();
    auto runtime = std::chrono::duration<double>(end - start).count();

    std::vector<double> R = residual(U, h, Y, X);
    double R_euclidean = euclidean_norm(R);
    double R_maximum   = maximum_norm(R);

    std::vector<double> T = totalerror(U, h, Y, X);
    double T_euclidean = euclidean_norm(T);
    double T_maximum   = maximum_norm(T);

    std::cout << std::scientific << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Resolution = " << resolution << std::endl;
    std::cout << "Iterations = " << iterations << std::endl;
    std::cout << "Runtime    = " << runtime << " seconds" << std::endl;
    std::cout << "Residual Euclidean Norm = " << R_euclidean << std::endl;
    std::cout << "Residual Maximum   Norm = " << R_maximum << std::endl;
    std::cout << "Totalerror Euclidean Norm = " << T_euclidean << std::endl;
    std::cout << "Totalerror Maximum   Norm = " << T_maximum << std::endl;
    std::cout << "========================================" << std::endl;
}
