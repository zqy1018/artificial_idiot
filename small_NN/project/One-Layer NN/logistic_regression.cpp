#include "logistic_regression.h"
#include <cmath>

Logistic_regression::Logistic_regression(int _targ): 
    param(IMAGE_SIZE + 1, 1, RANDOM){
    /* 初始化参数向量 */
    targ = _targ;
}

void Logistic_regression::fit(Mat_n learning_rate, int round){
    for (int i = 0; i < round; ++i){
        vector<vector<unsigned char> > images;
        vector<vector<Mat_n> > images_n;
        vector<unsigned char> labels;
        
        vector<int> indices;
        random_choice(55000, BATCH_SIZE, indices);
        pick_by_vector(
            "../train_data/train-images.idx3-ubyte", 
            "../train_data/train-labels.idx1-ubyte", 
            images,
            labels, 
            indices
        );

        /* 归一化 */
        normalization(images, images_n);

        /* 转换为正确的正类和负类 */
        for (int j = 0; j < BATCH_SIZE; ++j){
            if(labels[j] == targ) labels[j] = 1;
            else labels[j] = 0;
        }

        vector<Mat_n> res;
        vector<Mat_n> grad(IMAGE_SIZE + 1, 0);

        for (int j = 0; j < BATCH_SIZE; ++j){
            images_n[j].push_back(1);
            Matrix X(images_n[j]);
            Mat_n sca = sigmoid(Mat_n(param.T() ^ X));
            res.push_back(sca);
        }

        Mat_n J = loss(res, grad, images_n, labels);
        grad_descent(grad, learning_rate);
    }
}

Mat_n Logistic_regression::loss(const vector<Mat_n>& prediction, vector<Mat_n>& grad, vector<vector<Mat_n> >& images_n, 
    vector<unsigned char>& labels) const{
    /* 这里要求 grad 数组是全为0的 */
    Mat_n J = 0;
    for (int i = 0; i < BATCH_SIZE; ++i){
        if(labels[i]){
            J -= log(prediction[i]);
        }else{
            J -= log(1 - prediction[i]);
        }
        for (int j = 0; j <= IMAGE_SIZE; ++j)
            grad[j] += (prediction[i] - labels[i]) * images_n[i][j];   
    }
    J /= BATCH_SIZE;
    for (int i = 0; i <= IMAGE_SIZE; ++i)
        grad[i] /= BATCH_SIZE;
}

void Logistic_regression::grad_descent(const vector<Mat_n>& grad, Mat_n learning_rate){
    for (int i = 0; i <= IMAGE_SIZE; ++i)
        param[i][0] -= learning_rate * grad[i];
}

void Logistic_regression::predict(vector<Mat_n>& res){
    /* 这里要求 res 是空的 */
    vector<vector<unsigned char> > images;
    vector<vector<Mat_n> > images_n;
    vector<unsigned char> labels;

    read_examples(
        "../test_data/t10k-images.idx3-ubyte",
        "../test_data/t10k-labels.idx1-ubyte",
        images, 
        labels, 
        TEST_SIZE
    );
    normalization(images, images_n);

    for (int i = 0; i < TEST_SIZE; ++i){
        images_n[i].push_back(1);
        Matrix X(images_n[i]);
        Mat_n sca = sigmoid(Mat_n(param.T() ^ X));
        res.push_back(sca);
    }
}

Logistic_regression::~Logistic_regression(){
    ;
}