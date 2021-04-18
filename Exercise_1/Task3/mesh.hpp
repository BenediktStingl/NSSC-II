#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include "mpi.h"

template<typename T>
class Mesh
{
public:
    long N; /* Axis: x, Index: i, Interval: 0,...,N-1 - includes ghost layers */
    long M; /* Axis: y, Index: j, Interval: 0,...,M-1 - includes ghost layers */
    std::vector<T> values;

    long X; /* MPI_Cart_coords[1] */
    long Y; /* MPI_Cart_coords[0] */
    T h;

    int vert_src, vert_dest, hor_src, hor_dest; /* MPI_Cart_shift */
    long i, I; /* Axis: x, Domain-Index (without Boundaries), Interval: i,...,I */
    long j, J; /* Axis: y, Domain-Index (without Boundaries), Interval: j,...,J */
    long widthX; /* Axis: x, Domain-Width (without Boundaries) */
    long widthY; /* Axis: y, Domain-Width (without Boundaries) */

    std::vector<long> meshWidthsY;
    std::vector<long> meshWidthsX;
    
    std::string DIM;

    Mesh(long N, long M, long X, long Y, T h, 
        int vert_src, int vert_dest, int hor_src, int hor_dest, 
        std::vector<long> meshWidthsY, std::vector<long> meshWidthsX, std::string DIM):
        N(N), M(M), values(N*M, 0.0), X(X), Y(Y), h(h),
        vert_src(vert_src), vert_dest(vert_dest),  hor_src(hor_src), hor_dest(hor_dest), 
        meshWidthsY(meshWidthsY), meshWidthsX(meshWidthsX), DIM(DIM)
    {
        if(this->DIM == "1D")
        {
            this->i = 2; this->I = this->N-2; // horizontal Decomposition

            if(vert_src < 0 && vert_dest < 0) // one processor
            { this->j = 2; this->J = this->M-2; }
            else if(vert_src < 0) // if no source rank
            { this->j = 2; this->J = this->M-1; } 
            else if(vert_dest < 0) // if no dest rank
            { this->j = 1; this->J = this->M-2; }
            else 
            { this->j = 1; this->J = this->M-1; }

            this->widthX = I - i;
            this->widthY = J - j;
        }
        else // 2D decomposition
        { 
            if(vert_src < 0) // vertical - if no source rank
            { this->j = 2; this->J = this->M-1;}
            else if(vert_dest < 0) // if no dest rank
            { this->j = 1; this->J = this->M-2; }
            else 
            { this->j = 1; this->J = this->M-1; }
                    
            if(hor_src < 0) // horizontal - if no source rank
            { this->i = 2; this->I = this->N-1; }
            else if(hor_dest < 0)  // if no dest rank
            { this->i = 1; this->I = this->N-2; }
            else
            { this->i = 1; this->I = this->N-1; }
                    
            this->widthX = I - i;
            this->widthY = J - j;
        }

        /* Set Boundary Conditions */
        if(vert_dest < 0) // bottom only
        {
            for(long i = this->i; i < this->I; i++)
            {
                T value = sin( 2*M_PI*coord(i, 1, "x") ) * sinh( 2*M_PI );
                setValue(i, J, value);
            }
        }
    }

    long index(long i, long j)
    {
        return (N*j + i);
    }

    T getValue(long i, long j)
    {
        return this->values[index(i, j)]; 
    }

    void setValue(long i, long j, T value)
    {
        this->values[index(i, j)] = value; 
    }

    T coord(long i, long j, std::string axis)
    {
        T c = -1000;

        if(axis == "x") 
        { 
            long axisX = 0;
            for(long k = 0; k < X; k++) { axisX += this->meshWidthsX[k]; }
            c = (axisX)*this->h + (i-1)*(this->h); 
        }

        if(axis == "y") 
        {
            long axisY = 0;
            for(long k = 0; k < Y; k++) { axisY += this->meshWidthsY[k]; }
            c = (axisY)*this->h + (j-1)*(this->h); 
        }

        return c;
    }
    
    void GhostSwap()
    {
        if(this->vert_src >= 0) // top
        {
            std::vector<T> topBoundary(this->widthX, 0.0);
            for(long i = this->i; i < this->I; i++)
            {
                topBoundary[i-(this->i)] = getValue(i, this->j);
            }
            MPI_Send(&topBoundary.front(), topBoundary.size(), MPI_FLOAT, this->vert_src, 1, MPI_COMM_WORLD); //sends boundary

            std::vector<T> topGhost(this->widthX, 0.0);
            MPI_Recv(&topGhost.front(), topGhost.size(), MPI_FLOAT, this->vert_src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  //receives ghost
            for(long i = this->i; i < this->I; i++)
            {
                setValue(i, this->j-1, topGhost[i-this->i]);
            }
        }

        if(this->vert_dest >= 0) // bottom
        {
            std::vector<T> bottomBoundary(this->widthX, 0.0);
            for(long i = this->i; i < this->I; i++)
            {
                bottomBoundary[i-this->i] = getValue(i, this->J-1);
            }
            MPI_Send(&bottomBoundary.front(), bottomBoundary.size(), MPI_FLOAT, this->vert_dest, 2, MPI_COMM_WORLD); //sends boundary

            std::vector<T> bottomGhost(this->widthX, 0.0);
            MPI_Recv(&bottomGhost.front(), bottomGhost.size(), MPI_FLOAT, this->vert_dest, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  //receives ghost
            for(long i = this->i; i < this->I; i++)
            {
                setValue(i, this->J, bottomGhost[i-(this->i)]);
            }
        }
        
        if(this->DIM == "2D")
        {
            if(this->hor_src >= 0) // left
            {
                std::vector<T> leftBoundary(this->widthY, 0.0);
                for(long j = this->j; j < this->J; j++)
                {
                    leftBoundary[j-(this->j)] = getValue(this->i, j);
                }
                MPI_Send(&leftBoundary.front(), leftBoundary.size(), MPI_FLOAT, this->hor_src, 0, MPI_COMM_WORLD); //sends boundary

                std::vector<T> leftGhost(this->widthY, 0.0);
                MPI_Recv(&leftGhost.front(), leftGhost.size(), MPI_FLOAT, this->hor_src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //receives ghost
                for(long j = this->j; j < this->J; j++)
                {
                    setValue(this->i-1, j, leftGhost[j-this->j]);
                }
            }

            if(this->hor_dest >= 0) // right
            {
                std::vector<T> rightBoundary(this->widthY, 0.0);
                for(long j = this->j; j < this->J; j++)
                {
                    rightBoundary[j-this->j] = getValue(this->I-1, j);
                }
                MPI_Send(&rightBoundary.front(), rightBoundary.size(), MPI_FLOAT, this->hor_dest, 0, MPI_COMM_WORLD); //sends boundary

                std::vector<T> rightGhost(this->widthY, 0.0);
                MPI_Recv(&rightGhost.front(), rightGhost.size(), MPI_FLOAT, this->hor_dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //receives ghost
                for(long j = this->j; j < this->J; j++)
                {
                    setValue(this->I, j ,rightGhost[j-(this->j)]);
                }
            }
        }
    }
};
