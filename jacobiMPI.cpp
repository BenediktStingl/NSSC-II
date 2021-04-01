#include <unistd.h>
#include <math.h>
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

std::vector<long> splitMesh(long resolution, int NUMMPIPROC)//, int RANK)
{
    std::vector<long> meshWidths(NUMMPIPROC, 0);

    long width = resolution / NUMMPIPROC;

    for(long k = 0; k < NUMMPIPROC; k++)
    {
        meshWidths[k] = width;
        resolution = resolution - width;

        if(resolution < width || k == NUMMPIPROC-1)
        {
            meshWidths[k] = meshWidths[k] + resolution; //last one gets the residuals
            break;
        }
    }

    return meshWidths;
}

std::vector<long> distributeMeshWidths(long resolution, int NUMMPIPROC, int RANK, MPI_Status status, int all_ranks, int master)
{
    if(RANK == master)
    {
        std::vector<long> defaultMeshWidths = splitMesh(resolution, NUMMPIPROC);//, RANK);
        for(long k = 0; k < all_ranks; k++)
        {
            MPI_Send(&defaultMeshWidths.front(), defaultMeshWidths.size(), MPI_LONG, k, 0, MPI_COMM_WORLD); //broadcast here?
        }
    }
    
    std::vector<long> meshWidths(NUMMPIPROC, 0);
    MPI_Recv(&meshWidths.front(), meshWidths.size(), MPI_LONG, master, 0, MPI_COMM_WORLD, &status);
    
    return meshWidths;
}

int PrimeDetector(int NUMMPIPROC)
{
    double div_temp;
    int div = 0, l = 2; //l = 1 is always divisible

    while (l <= round(sqrt(NUMMPIPROC)))
        {
            div_temp = NUMMPIPROC % l; //if divisible = 0
            if (div_temp == 0) 
            {
                div = l; //makes this the largest divisor found so far
            } 
            l++; 
        }
         return div;
}

std::vector<double> Receive_vectors1D(int NUMMPIPROC, long meshWidthX, std::vector<long> meshWidthsY, std::vector<double>V, MPI_Status status, int tag)
{
    std::vector<double> V_global = V;
    for(long k = 1; k < NUMMPIPROC; k++)
    {
        long length = (meshWidthX+2) * (meshWidthsY[k]+2);
        std::vector<double> V_tmp(length, 0.0);
        
        MPI_Recv(&V_tmp.front(), length, MPI_DOUBLE, k, tag, MPI_COMM_WORLD, &status);
        V_global.insert(V_global.end(), V_tmp.begin(), V_tmp.end());
    }
    return V_global;
}

std::vector<double> Receive_vectors2D(int NUMMPIPROC, std::vector<long> meshWidthsX, std::vector<long> meshWidthsY, std::vector<double>V, MPI_Status status, int tag, int len_meshX, int len_meshY)
{
    std::vector<double> V_global = V;
    
    for(long ky = 0; ky < len_meshY; ky++)
    {
        for(long kx = 0; kx < len_meshX; kx++)
        {
            if (kx ==0 && ky==0)
            {
                continue;
            }
            long length = (meshWidthsX[kx]+2) * (meshWidthsY[ky]+2);
            std::vector<double> V_tmp(length, 0.0);
            
            MPI_Recv(&V_tmp.front(), length, MPI_DOUBLE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status); //MPI_Recv(&V_tmp.front(), length, MPI_DOUBLE, ky*len_meshX+kx, tag, MPI_COMM_WORLD, &status);
            V_global.insert(V_global.end(), V_tmp.begin(), V_tmp.end());
        }
    } 
    return V_global;
}


int main(int argc, char* argv[])
{
    char* decomp = argv[1];
    long resolutionY = read_parameter(2, argc, argv, 50);
    long iterations  = read_parameter(3, argc, argv, 500);
    
    long resolutionX = 2*resolutionY - 1;
    double h = 1.0/(resolutionY - 1);

    int RANK, NUMMPIPROC, factor = 0, master = 0;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &NUMMPIPROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Status status;
    
    if (decomp[0]=='2') //not beautiful....
    {
        if (RANK == master)
        {
             factor = PrimeDetector(NUMMPIPROC); //find dividing factor for 2D decomposition
        }
        MPI_Bcast(&factor, 1, MPI_INT, master, MPI_COMM_WORLD); //send factor to all ranks
    }
    
     if (decomp[0]=='1' || factor == 0) //factor ==0 means its a prime number 
         {//executes 1D decomposition
         
         MPI_Comm cartTopology;
        int ndims = 1;
        int dims[2] = {};
        int periods[2] = {0,0}; 
        int reorder = 0;
        MPI_Dims_create(NUMMPIPROC, ndims, dims);
        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cartTopology);

        int coords[2] = {};
        MPI_Cart_coords(cartTopology, RANK, ndims, coords);

        std::vector<long> meshWidthsY = distributeMeshWidths(resolutionY, NUMMPIPROC, RANK, status, NUMMPIPROC, master);
        std::vector<long> meshWidthsX(NUMMPIPROC, 0);
        long meshWidthX = resolutionX;
        long meshWidthY = meshWidthsY[RANK];

        int vert_src, vert_dest, hor_src=-10, hor_dest=-10;
        MPI_Cart_shift(cartTopology, 0, 1, &vert_src, &vert_dest);

        Mesh<double> mesh(meshWidthX+2, meshWidthY+2, coords[1], coords[0], h,
                          vert_src, vert_dest, hor_src, hor_dest, meshWidthsY, meshWidthsX);

        mesh.setBoundaryConditions();

        mesh.fill(0.0); //initializes domain to 0
    
        for(long k = 0; k < iterations; k++)
        {
            jacobi(mesh);
            mesh.verticalGhostSwap(status);
        }

        std::vector<double> R = residual(mesh);
        std::vector<double> T = totalerror(mesh);
        int tagR=0, tagT =1;
        
        if(RANK != master)
        {
            MPI_Send(&R.front(), R.size(), MPI_DOUBLE, master, tagR, MPI_COMM_WORLD);
            MPI_Send(&T.front(), T.size(), MPI_DOUBLE, master, tagT, MPI_COMM_WORLD);
        }
        else 
        {
            std::vector<double> R_global;
            std::vector<double> T_global;
            
            R_global = Receive_vectors1D(NUMMPIPROC, meshWidthX, meshWidthsY, R, status, tagR);
            T_global = Receive_vectors1D(NUMMPIPROC, meshWidthX, meshWidthsY, T, status, tagT);

            double R_E = euclidean_norm(R_global);
            double R_M = maximum_norm(R_global);
            double T_E = euclidean_norm(T_global);
            double T_M = maximum_norm(T_global);

            std::cout << std::scientific << std::endl;
            std::cout << "R_E = " << R_E << std::endl;
            std::cout << "R_M = " << R_M << std::endl;
            std::cout << "T_E = " << T_E << std::endl;
            std::cout << "T_M = " << T_M << std::endl;
        } 
     }
     else 
     {//executes 2D decomposition
         
         MPI_Comm cartTopology;
        int ndims = 2; //2D decomp
        int dims[2] = {factor, NUMMPIPROC/factor};
        int periods[2] = {0,0}; 
        int reorder = 0;
        MPI_Dims_create(NUMMPIPROC, ndims, dims);
        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cartTopology);

        int coords[2] = {};
        MPI_Cart_coords(cartTopology, RANK, ndims, coords);

        int len_meshX = NUMMPIPROC/factor, len_meshY = factor;
        std::vector<long> meshWidthsY = distributeMeshWidths(resolutionY, len_meshY, RANK, status, NUMMPIPROC, master);
        std::vector<long> meshWidthsX = distributeMeshWidths(resolutionX, len_meshX, RANK, status, NUMMPIPROC, master);

        long meshWidthX = meshWidthsX[coords[1]];
        long meshWidthY = meshWidthsY[coords[0]];

        int vert_src, vert_dest, hor_src, hor_dest;
        MPI_Cart_shift(cartTopology, 0, 1, &vert_src, &vert_dest);
        MPI_Cart_shift(cartTopology, 1, 1, &hor_src, &hor_dest);

        Mesh<double> mesh(meshWidthX+2, meshWidthY+2, coords[1], coords[0], h,
                          vert_src, vert_dest, hor_src, hor_dest, meshWidthsY, meshWidthsX);

        mesh.setBoundaryConditions();

        mesh.fill(0.0);
        for(long k = 0; k < iterations; k++)
        {
            jacobi(mesh);
            mesh.GhostSwap(status);
        }
        std::vector<double> R = residual(mesh);
        std::vector<double> T = totalerror(mesh);
        int tagR=0, tagT =1;
        
        if(RANK != master)
        {
            MPI_Send(&R.front(), R.size(), MPI_DOUBLE, master, tagR, MPI_COMM_WORLD);
            MPI_Send(&T.front(), T.size(), MPI_DOUBLE, master, tagT, MPI_COMM_WORLD);
        }
        else 
        {
            std::vector<double> R_global;
            std::vector<double> T_global;
            
            R_global=Receive_vectors2D(NUMMPIPROC, meshWidthsX, meshWidthsY, R, status, tagR, len_meshX, len_meshY);
            T_global=Receive_vectors2D(NUMMPIPROC, meshWidthsX, meshWidthsY, T, status, tagT, len_meshX, len_meshY);

            double R_E = euclidean_norm(R_global);
            double R_M = maximum_norm(R_global);
            double T_E = euclidean_norm(T_global);
            double T_M = maximum_norm(T_global);

            std::cout << std::scientific << std::endl;
            std::cout << "R_E = " << R_E << std::endl;
            std::cout << "R_M = " << R_M << std::endl;
            std::cout << "T_E = " << T_E << std::endl;
            std::cout << "T_M = " << T_M << std::endl;
        } 
     }
    MPI_Finalize();
    return 0;
}
