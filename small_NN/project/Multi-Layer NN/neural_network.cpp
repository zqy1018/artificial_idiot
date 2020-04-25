#include "neural_network.h"
#include <cmath>

Matrix sigmoid_grad(const Matrix& m1){
    return sigmoid(m1) * (1 - sigmoid(m1));
}

Neural_network::Neural_network(): 
    param1(IMAGE_SIZE + 1, HIDDEN_LAYER_UNIT, RANDOM), 
    param2(HIDDEN_LAYER_UNIT + 1, 10, RANDOM){
    /* 初始化参数矩阵 */
    ;
}

void Neural_network::fit(Mat_n learning_rate, int round){
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

        Matrix D1(param1.getRow(), param1.getColumn(), ALL_ZERO);
        Matrix D2(param2.getRow(), param2.getColumn(), ALL_ZERO);

        /* 归一化 */
        normalization(images, images_n);

        for (int j = 0; j < BATCH_SIZE; ++j){
            images_n[j].push_back(1);
            Matrix X(images_n[j]);

            /* 前向传播 */
            Matrix z2 = param1.T() ^ X;
            Matrix a2(z2.getRow() + 1, 1, ALL_ZERO);
            for (int k = 0; k < z2.getRow(); ++k)
                a2[k][0] = sigmoid(z2[k][0]);
            a2[z2.getRow()][0] = 1;

            Matrix z3 = param2.T() ^ a2;
            Matrix a3 = sigmoid(z3);

            /* 获取标签 */
            vector<Mat_n> labels_n(10, 0);
            labels_n[labels[j]] = 1;

            /* 反向传播 */
            Matrix delta3 = (a3 - Matrix(labels_n));
            Matrix cap_delta3 = delta3 ^ a2.T();
            D2 = D2 + cap_delta3.T();

            Matrix delta2 = ((param2 ^ delta3) * (a2 * (1 - a2)));
            Matrix delta2_n(delta2.getRow() - 1, 1, ALL_ZERO);
            for (int k = 0; k < delta2_n.getRow(); ++k)
                delta2_n[k][0] = delta2[k][0];
            Matrix cap_delta2 = delta2_n ^ X.T();
            D1 = D1 + cap_delta2.T();
        }

        grad_descent(D1 / BATCH_SIZE, D2 / BATCH_SIZE, learning_rate);
    }
}

void Neural_network::grad_descent(const Matrix& grad1, const Matrix& grad2, Mat_n learning_rate){
    param1 = param1 - (learning_rate * grad1);
    param2 = param2 - (learning_rate * grad2);
}

void Neural_network::predict(vector<int>& res){
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
        
        /*  前向传播 */
        Matrix z2 = param1.T() ^ X;
        Matrix a2(z2.getRow() + 1, 1, ALL_ZERO);
        for (int j = 0; j < z2.getRow(); ++j)
            a2[j][0] = sigmoid(z2[j][0]);
        a2[z2.getRow()][0] = 1;

        Matrix z3 = param2.T() ^ a2;
        Matrix a3 = sigmoid(z3);

        int maxi = -1;
        Mat_n cur = -10;
        for (int j = 0; j < 10; ++j)
            if(cur < a3[j][0])
                cur = a3[j][0], maxi = j;
        res[i] = maxi;
    }
}

Neural_network::~Neural_network(){
    ;
}