#include <fstream>
#include <iostream>
#include <string>   
#include <vector>
#include <stdio.h>
#include <sstream>
#include <math.h>
using namespace std;
int main(){
    string fileName = "/home/avtimofeev/opendust/examples/DustDynamics/TwoParticles/data/trajectory.xyz";
    string outputFileName = "/home/avtimofeev/opendust/examples/IonConcentration/data/concentration.txt";
    ifstream infile(fileName);    
    string line;
    double *x;
    double *y; 
    double *z;
    string element;
    int counter = 0;
    int N;
    int n = 100;
    double z_grid[n];
    double N_s = 63.95267339533143;
    double R = 0.0014940882723609578;
    vector<vector<double>> C_array;
    vector<double> C;
    for (int index = 0; index < n; index++)
    {
        C.push_back(0);
    }
    double L = 0.005976353089443831;
    
    for (int i = 0; i < n; i++){
        z_grid[i] = -L/2.0 + i*L/(double)(n-1);
    }
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        
        if (counter == 0){
            std::istringstream iss(line);
            iss >> N;
            counter ++;
            x = new double[N];
            y = new double[N];
            z = new double[N];

        } else if (counter == 1){
            counter ++;
            for (int index = 0; index < n; index++)
            {
                C[index] = 0;
            }
        } else {
            std::istringstream iss(line);
            iss >> element >> x[counter-2] >> y[counter-2] >> z[counter-2];
            x[counter-2] /= 1e9;
            y[counter-2] /= 1e9;
            z[counter-2] /= 1e9;
            int index = (int)((0.5*L+z[counter-2]) * (n - 1) / L);


            if ((index == 0) && (z[counter-2] >= -0.5*L))
            {
                C[1] += 1.0 - (z_grid[1] - z[counter-2]) / (z_grid[1] - z_grid[0]);
            }
            else if (index == n - 2)
            {
                C[n - 2] += 1.0 - (z[counter-2] - z_grid[n - 2]) / (z_grid[n - 1] - z_grid[n - 2]);
            }
            else if ((index < n - 2) && (index > 0))
            {
                C[index] += 1.0 - (z[counter-2] - z_grid[index]) / (z_grid[index + 1] - z_grid[index]);
                C[index + 1] += 1.0 - (z_grid[index + 1] - z[counter-2]) / (z_grid[index + 1] - z_grid[index]);
            }
            counter ++;
            if (counter == N+2){
                counter = 0;
                
                for (int index = 0; index < n; index++)
                {
                    if ((index < n - 1) && (index > 0))
                    {
                        C[index] /= 0.5 * ((z_grid[index + 1] - z_grid[index]) + (z_grid[index] - z_grid[index - 1]))*(M_PI*R*R)/N_s;
                    }
                }
                C[0] = 2 * C[1] - C[2];
                C[n - 1] = 2 * C[n - 2] - C[n - 3];
                C_array.push_back(C);
                delete [] x;
                delete [] y;
                delete [] z;
            }
        }
    }
    
    for (int i = 0; i < C_array.size(); i++){
        for(int index = 0; index < n; index++){
            printf("%f\t%f\n",z_grid[index], C_array[i][index]);
        }
        printf("\n");
        printf("\n");
    }
    infile.close();
    return 0;
}