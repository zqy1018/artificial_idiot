#include "neural_network.h"
#include <iostream>
#include <ctime>
using namespace std;
int main(){
    clock_t start, end;

    Neural_network nn;
    
    start = clock();
    nn.fit(0.1, 300);
    end = clock();

    vector<int> pre_res(TEST_SIZE, 0);
    nn.predict(pre_res);

    vector<int> TP(10, 0), TN(10, 0), FP(10, 0), FN(10, 0);
    compare(
        "../test_data/t10k-labels.idx1-ubyte",
        pre_res, 
        TP, 
        TN, 
        FP, 
        FN
    );

    /* 显示结果 */
    cout << "Training Time: " << (end - start) << "ms" << endl;

    for (int i = 0; i < 10; ++i){
        cout << "For index " << i << ": " << endl;
        double precis = 1.0 * TP[i] / (TP[i] + FP[i]);
        double recall = 1.0 * TP[i] / (TP[i] + FN[i]);
        cout << "F1 score: " << 2.0 * precis * recall / (precis + recall) << endl;
    }

    return 0;
}
