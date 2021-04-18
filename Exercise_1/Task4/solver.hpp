#include "mesh.hpp"

template<typename T>
void jacobi(Mesh<T>& mesh)
{
    std::vector<T> U_k = mesh.values;

    for(long i = mesh.i; i < mesh.I; i++)
    {
        for(long j = mesh.j; j < mesh.J; j++)
        {
            double F = 4 * M_PI*M_PI * mesh.h*mesh.h;
            double RHS = F * sin( 2*M_PI*mesh.coord(i, j, "x") ) * sinh( 2*M_PI*mesh.coord(i, j, "y") );
            double SUM = (-1)*mesh.getValue(i, j-1) + (-1)*mesh.getValue(i-1, j) + (-1)*mesh.getValue(i, j+1) + (-1)*mesh.getValue(i+1, j);
            U_k[mesh.index(i, j)] = (RHS - SUM) / (4 + F);
        }
    }

    mesh.values = U_k;
}

template<typename T>
std::vector<T> residual(Mesh<T> mesh)
{
    /* Calculate R */
    std::vector<T> R(mesh.N*mesh.M, 0.0);

    for(long i = mesh.i; i < mesh.I; i++)
    {
        for(long j = mesh.j; j < mesh.J; j++)
        {
            T F = 4 * M_PI*M_PI * mesh.h*mesh.h;
            T RHS = F * sin( 2*M_PI*mesh.coord(i, j, "x") ) * sinh( 2*M_PI*mesh.coord(i, j, "y") );
            T LHS = (-1)*mesh.getValue(i, j-1) + (-1)*mesh.getValue(i-1, j) + (4+F)*mesh.getValue(i, j) + (-1)*mesh.getValue(i, j+1) + (-1)*mesh.getValue(i+1, j);
            R[mesh.index(i, j)] = LHS - RHS;
        }
    }

    /* Transform R */
    std::vector<T> R_trans(mesh.widthX*mesh.widthY, 0.0);

    for(long i = mesh.i; i < mesh.I; i++)
    {
        for(long j = mesh.j; j < mesh.J; j++)
        {
            long Index = mesh.widthX*(j-mesh.j) + (i-mesh.i);
            R_trans[Index] = R[mesh.index(i, j)];
        }
    }
    
    return R_trans;
}

template<typename T>
std::vector<T> solution(Mesh<T> mesh)
{
    std::vector<T> S(mesh.N*mesh.M, 0.0);

    for(long i = mesh.i; i < mesh.I; i++)
    {
        for(long j = mesh.j; j < mesh.J; j++)
        {
            S[mesh.index(i, j)] = sin( 2*M_PI*mesh.coord(i, j, "x") ) * sinh( 2*M_PI*mesh.coord(i, j, "y") );
        }
    }

    return S;
}

template<typename T>
std::vector<T> totalerror(Mesh<T> mesh)
{ 
    /* Calculate Total */
    std::vector<T> Total(mesh.N*mesh.M, 0.0);
    std::vector<T> S = solution(mesh);

    for(long i = mesh.i; i < mesh.I; i++)
    {
        for(long j = mesh.j; j < mesh.J; j++)
        {
            Total[mesh.index(i, j)] = mesh.getValue(i, j) - S[mesh.index(i,j)];
        }
    }

    /* Transform Total */
    std::vector<T> Total_trans(mesh.widthX*mesh.widthY, 0.0);

    for(long i = mesh.i; i < mesh.I; i++)
    {
        for(long j = mesh.j; j < mesh.J; j++)
        {
            long Index = mesh.widthX*(j-mesh.j) + (i-mesh.i);
            Total_trans[Index] = Total[mesh.index(i, j)];
        }
    }

    return Total_trans;
}

template<typename T>
T euclidean_norm(std::vector<T> V)
{
    T euclidean = 0;
    for (T element : V)
    {
        euclidean += element*element;
    }
    return sqrt(euclidean) / V.size();
}

template<typename T>
T maximum_norm(std::vector<T> V)
{
    T maximum = 0;
    for (T element : V)
    {
        maximum = std::fabs(element) > maximum ? std::fabs(element) : maximum;
    }
    return maximum;
}

