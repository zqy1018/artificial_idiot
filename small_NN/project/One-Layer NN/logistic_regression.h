#ifndef LOGISTIC

    #define LOGISTIC

    #include "../matrix.h"
    #include "../training.h"

    class Logistic_regression{
    private:
        int targ; // 当前回归的目标标签
        Matrix param;
        Mat_n loss(const vector<Mat_n>& prediction, vector<Mat_n>& grad, vector<vector<Mat_n> >& images_n, 
            vector<unsigned char>& labels) const;
        void grad_descent(const vector<Mat_n>& grad, Mat_n learning_rate);
    public:
        explicit Logistic_regression(int _targ);
        void fit(Mat_n learning_rate, int round = 100);
        void predict(vector<Mat_n>& res);
        ~Logistic_regression();
    };

#endif