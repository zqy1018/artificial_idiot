#include "logistic_regression.h"
#include <iostream>
#include <ctime>
using namespace std;
int main(){
    clock_t start, end;

    Logistic_regression *classifier[10];
    for (int i = 0; i < 10; ++i)
        classifier[i] = new Logistic_regression(i);
    
    start = clock();
    for (int i = 0; i < 10; ++i)
        classifier[i]->fit(0.2, 100);
    end = clock();
    
    vector<vector<Mat_n> > res(10, vector<Mat_n>());
    for (int i = 0; i < 10; ++i){
        classifier[i]->predict(res[i]);
    }

    vector<int> pre_res(TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; ++i){
        int maxi = -1;
        Mat_n cur = -10;
        for (int j = 0; j < 10; ++j)
            if(cur < res[j][i])
                cur = res[j][i], maxi = j;
        pre_res[i] = maxi;
    }

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
