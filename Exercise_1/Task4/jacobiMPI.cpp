#include <unistd.h>
#include <math.h>
#include <chrono>
#include <iomanip>
#include "solver.hpp"

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

int PrimeDetector(int NUMMPIPROC)
{
    double div_temp;
    int div = 0, l = 2; // l = 1 is always divisible

    while (l <= round(sqrt(NUMMPIPROC)))
        {
            div_temp = NUMMPIPROC % l; // if divisible = 0
            if (div_temp == 0) 
            {
                div = l; // makes this the largest divisor found so far
            } 
            l++; 
        }
         return div;
}

std::vector<long> splitMesh(long resolution, int NUMMPIPROC)
{
    std::vector<long> meshWidths(NUMMPIPROC, 0);

    long width = resolution / NUMMPIPROC;

    for(long k = 0; k < NUMMPIPROC; k++)
    {
        meshWidths[k] = width;
        resolution = resolution - width;

        if(resolution < width || k == NUMMPIPROC-1)
        {
            meshWidths[k] = meshWidths[k] + resolution; // last one gets the residuals
            break;
        }
    }

    return meshWidths;
}

template<typename T>
std::vector<T> Receive_vectors(int NUMMPIPROC, std::vector<T>& V, int tag)
{
    for(long k = 1; k < NUMMPIPROC; k++)
    {
        long length = 0;
        MPI_Recv(&length, 1, MPI_LONG, k, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<T> V_tmp(length, 0.0);
        
        MPI_Recv(&V_tmp.front(), length, MPI_DOUBLE, k, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        V.insert(V.end(), V_tmp.begin(), V_tmp.end());
    }
    return V;
}

std::string Decomp_from_Input(std::string dim, int factor)
{
    std::string DIM;
    
    if(dim == "2D" && factor != 0) //factor == 0 means it is a prime number
        DIM = "2D";
    else if(dim == "1D" || factor == 0)
        DIM = "1D";
    else //in case a wrong input is provided also switch to 1D decomp
    {
        DIM = "1D";
        std::cout << "Input not understood - changed decomposition to 1D." << std::endl;
    }
    
    return DIM;
}

template<typename Type>
void calculation(int RANK, int NUMMPIPROC, Mesh<Type> mesh, int master, long iterations, int resolution, std::string DIM)
{
    auto start     = std::chrono::steady_clock::now();
    for(long k = 0; k < iterations; k++)
    {
        jacobi<Type>(mesh);
        mesh.GhostSwap();
    }
    auto end       = std::chrono::steady_clock::now();
    auto runtime   = std::chrono::duration<Type>(end - start).count();
    Type average = runtime / iterations;

    std::vector<Type> R = residual(mesh);
    std::vector<Type> T = totalerror(mesh);
    int tagR = 0, tagT = 1;
    
    if(RANK != master)
    {
        long vectorLength_R = mesh.widthX*mesh.widthY;
        MPI_Send(&vectorLength_R, 1, MPI_LONG, master, tagR, MPI_COMM_WORLD);
        MPI_Send(&R.front(), R.size(), MPI_DOUBLE, master, tagR, MPI_COMM_WORLD);

        long vectorLength_T = mesh.widthX*mesh.widthY;
        MPI_Send(&vectorLength_T, 1, MPI_LONG, master, tagT, MPI_COMM_WORLD);
        MPI_Send(&T.front(), T.size(), MPI_DOUBLE, master, tagT, MPI_COMM_WORLD);
    }
    else 
    {
        std::vector<Type> R_global;
        std::vector<Type> T_global;
        
        R_global = Receive_vectors(NUMMPIPROC, R, tagR);
        T_global = Receive_vectors(NUMMPIPROC, T, tagT);

        std::cout << std::fixed;
        std::cout << std::setprecision(8);
        std::cout << std::scientific << std::endl;
		std::cout << "No. of processors: " << NUMMPIPROC << " resolution: " << resolution << " iterations: " << iterations << " DIM: " << DIM << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total   Runtime = " << runtime << " seconds" << std::endl;
        std::cout << "Average Runtime = " << average << " seconds" << std::endl;
        std::cout << "Residual Euclidean Norm = " << euclidean_norm(R_global) << std::endl;
        std::cout << "Residual Maximum   Norm = " << maximum_norm(R_global)   << std::endl;
        std::cout << "Totalerror Euclidean Norm = " << euclidean_norm(T_global) << std::endl;
        std::cout << "Totalerror Maximum   Norm = " << maximum_norm(T_global)   << std::endl;
        std::cout << "========================================" << std::endl;
    } 
}


int main(int argc, char* argv[])
{
    long resolutionY = read_parameter(2, argc, argv, 50);
    long iterations  = read_parameter(3, argc, argv, 500);
    
    long resolutionX = 2*resolutionY - 1;
    double h = 1.0/(resolutionY - 1);

    MPI_Init(&argc, &argv);
    int RANK, NUMMPIPROC;
    MPI_Comm_size(MPI_COMM_WORLD, &NUMMPIPROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

    int master = 0;
    int factor = PrimeDetector(NUMMPIPROC); // find dividing factor for 2D decomposition/if it is a prime number
    std::string DIM  = Decomp_from_Input(argv[1], factor);

    if(DIM == "1D")
    { // executes 1D decomposition
         
        MPI_Comm cartTopology;
        int ndims = 1;
        int dims[2] = {};
        int periods[2] = {0,0}; 
        int reorder = 0;
        MPI_Dims_create(NUMMPIPROC, ndims, dims);
        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cartTopology);

        int coords[2] = {};
        MPI_Cart_coords(cartTopology, RANK, ndims, coords);

        std::vector<long> meshWidthsX(NUMMPIPROC, 0);
        std::vector<long> meshWidthsY = splitMesh(resolutionY, NUMMPIPROC);
        long meshWidthX = resolutionX;
        long meshWidthY = meshWidthsY[RANK];

        int vert_src, vert_dest, hor_src=-10, hor_dest=-10;
        MPI_Cart_shift(cartTopology, 0, 1, &vert_src, &vert_dest);

        Mesh<double> mesh(meshWidthX+2, meshWidthY+2, coords[1], coords[0], h,
                          vert_src, vert_dest, hor_src, hor_dest, meshWidthsY, meshWidthsX, DIM);
        
        calculation<double>(RANK, NUMMPIPROC, mesh, master, iterations, resolutionY, DIM);

    }
    else
    { // executes 2D decomposition
         
        MPI_Comm cartTopology;
        int ndims = 2;
        int dims[2] = {factor, NUMMPIPROC/factor};
        int periods[2] = {0,0}; 
        int reorder = 0;
        MPI_Dims_create(NUMMPIPROC, ndims, dims);
        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cartTopology);

        int coords[2] = {};
        MPI_Cart_coords(cartTopology, RANK, ndims, coords);

        int len_meshX = NUMMPIPROC/factor, len_meshY = factor;
        std::vector<long> meshWidthsX = splitMesh(resolutionX, len_meshX);
        std::vector<long> meshWidthsY = splitMesh(resolutionY, len_meshY);
        long meshWidthX = meshWidthsX[coords[1]];
        long meshWidthY = meshWidthsY[coords[0]];

        int vert_src, vert_dest, hor_src, hor_dest;
        MPI_Cart_shift(cartTopology, 0, 1, &vert_src, &vert_dest);
        MPI_Cart_shift(cartTopology, 1, 1, &hor_src, &hor_dest);

        Mesh<double> mesh(meshWidthX+2, meshWidthY+2, coords[1], coords[0], h,
                          vert_src, vert_dest, hor_src, hor_dest, meshWidthsY, meshWidthsX, DIM);
        
        calculation<double>(RANK, NUMMPIPROC, mesh, master, iterations, resolutionY, DIM);

    }

    MPI_Finalize();
    return 0;
}
