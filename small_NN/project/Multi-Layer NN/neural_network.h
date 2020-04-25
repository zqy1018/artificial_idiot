#ifndef NEURAL

    #define NEURAL

    #include "../matrix.h"
    #include "../training.h"

    const int HIDDEN_LAYER_UNIT = 40;

    Matrix sigmoid_grad(const Matrix& m1);

    class Neural_network{
    private:
        Matrix param1, param2;
        void grad_descent(const Matrix& grad1, const Matrix& grad2, Mat_n learning_rate);
    public:
        Neural_network();
        void fit(Mat_n learning_rate, int round = 100);
        void predict(vector<int>& res);
        ~Neural_network();
    };

#endif