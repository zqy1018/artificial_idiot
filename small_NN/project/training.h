#ifndef MY_TRAINING

    #define MY_TRAINING

    #include <vector>
    #include "matrix.h"
    using std::vector;

    const int IMAGE_SIZE = 28 * 28;
    /* 图片的大小 */

    const int BATCH_SIZE = 2000;
    /* 使用mini_batch训练方法时，一个batch的大小 */

    const int TEST_SIZE = 2000;
    /* 最终测试集的大小 */

    void read_examples(const char* image_name, const char* label_name, 
        vector<vector<unsigned char>>& vec_image, vector<unsigned char>& vec_label, int cnt = 1000, int offset = 0);
    /* 读取各种用例 */

    Mat_n sigmoid(Mat_n x);

    Matrix sigmoid(const Matrix& m1);

    void random_choice(int range, int cnt, vector<int>& res);
    /* 选取[0, range) 区间内的 cnt 个数 */

    void pick_by_vector(const char* image_name, const char* label_name, 
        vector<vector<unsigned char>>& vec_image, vector<unsigned char>& vec_label, const vector<int>& res);
    
    void normalization(const vector<vector<unsigned char> >& vec_image, vector<vector<Mat_n> >& res);

    void compare(const char* label_name, const vector<int>& pre_res, vector<int>& TP, 
        vector<int>& TN, vector<int>& FP, vector<int>& FN);

#endif