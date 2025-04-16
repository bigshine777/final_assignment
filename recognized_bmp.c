#include "nn.h"

int main(int argc, char * argv[]){
    float *A1 = malloc(sizeof(float)*784*50);
    float *b1 = malloc(sizeof(float)*50);
    float *A2 = malloc(sizeof(float)*50*100);
    float *b2 = malloc(sizeof(float)*100);
    float *A3 = malloc(sizeof(float)*100*10);
    float *b3 = malloc(sizeof(float)*10);
    
    float *x = load_mnist_bmp(argv[4]);
    
    load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);
    

    return 0;
}