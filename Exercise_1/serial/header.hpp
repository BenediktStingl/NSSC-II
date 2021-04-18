#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

long read_parameter(long position, int argc, char *argv[], long Default)
{
    if (argc <= position)
    {
        std::cout << "Parameter " << position <<
                     " not existent. Using default instead." << std::endl;
        return Default;
    }

    long parameter;
    std::istringstream tmp(argv[position]);
    if ( !(tmp >> parameter) )
    {
        std::cout << "Parameter "  << position << 
                     " not an integer. Using default instead." << std::endl;
        return Default;
    }

    return parameter;
}

std::vector<double> generate_U(long Y, long X, double h)
{
    std::vector<double> U(Y*X, 0.0);
    for (long x = 0, i = (Y-1)*X; i < Y*X; x++, i++)
    {
        U[i] = sin( 2*M_PI*(x*h) ) * sinh( 2*M_PI );
    }
    return U;
}

std::vector<double> residual(std::vector<double> U, double h, long Y, long X)
{
    std::vector<double> R((Y-2)*(X-2), 0.0); 
    for (long y = 0, i = 1; i < Y-1; y++, i++)
    {
        for (long x = 0, j = 1; j < X-1; x++, j++)
        {
            long   I = X*i + j;
            double F = (4 * M_PI*M_PI * h*h);
            double RHS = F * sin( 2*M_PI*(j*h) ) * sinh( 2*M_PI*(i*h) );
            double LHS = (-1)*U[I-1] + (-1)*U[I-X] + (4+F)*U[I] + (-1)*U[I+1] + (-1)*U[I+X];
            R[(X-2)*y + x] = LHS - RHS;
        }
    }
    return R;
}

std::vector<double> generate_S(long Y, long X, double h)
{
    std::vector<double> S(Y*X, 0.0);
    for (long i = 0; i < Y; i++)
    {
        for (long j = 0; j < X; j++)
        {
            S[X*i + j] = sin( 2*M_PI*(j*h) ) * sinh( 2*M_PI*(i*h) );
        }
    }
    return S;
}

std::vector<double> totalerror(std::vector<double> U, double h, long Y, long X)
{
    std::vector<double> S = generate_S(Y, X, h);
    std::vector<double> T(Y*X, 0.0);
    for (long i = 0; i < Y; i++)
    {
        for (long j = 0; j < X; j++)
        {
            T[X*i + j] = U[X*i + j] - S[X*i + j];
        }
    }

    
    /* Transform T*/
    std::vector<double> T_trans((Y-2)*(X-2), 0.0); 
    for (long y = 0, i = 1; i < Y-1; y++, i++)
    {
        for (long x = 0, j = 1; j < X-1; x++, j++)
        {
            T_trans[(X-2)*y + x] = T[X*i + j];
        }
    }

    return T_trans;
}

double euclidean_norm(std::vector<double> V)
{
    double euclidean = 0;
    for (double element : V)
    {
        euclidean += element*element;
    }
    return sqrt(euclidean) / V.size(); /* Normalization */
}

double maximum_norm(std::vector<double> V)
{
    double maximum = 0;
    for (double element : V)
    {
        maximum = std::fabs(element) > maximum ? std::fabs(element) : maximum;
    }
    return maximum;
}
