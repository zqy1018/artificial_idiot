#include "training.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>

void read_examples(const char* image_name, const char* label_name, vector<vector<unsigned char>>& vec_image, 
    vector<unsigned char>& vec_label, int cnt, int offset){
    FILE* in_image = fopen(image_name, "rb");
    FILE* in_label = fopen(label_name, "rb");

    fseek(in_image, 16L + offset * IMAGE_SIZE, SEEK_SET);
    fseek(in_label, 8L + offset, SEEK_SET);

    for (int i = 0; i < cnt; ++i){
        vector<unsigned char> cur_image(IMAGE_SIZE, 0);
        unsigned char cur_label = 0;

        for (int j = 0; j < IMAGE_SIZE; ++j)    
            fread(&cur_image[j], 1, 1, in_image);
        fread(&cur_label, 1, 1, in_label);

        vec_label.push_back(cur_label);
        vec_image.push_back(cur_image);
    }

    fclose(in_image);
    fclose(in_label);
}

Mat_n sigmoid(Mat_n x){
    return 1 / (1 + exp(-x));
}

Matrix sigmoid(const Matrix& m1){
    Matrix res(m1.getRow(), m1.getColumn(), ALL_ZERO);
    int r = m1.getRow(), c = m1.getColumn();
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            res[i][j] = sigmoid(m1[i][j]);
    return res;
}

void random_choice(int range, int cnt, vector<int>& res){
    srand(time(NULL));
    for (int i = 0; i < cnt; ++i){
        int choice = (rand() * (RAND_MAX + 1) + rand()) % range;
        res.push_back(choice);
    }
}

void pick_by_vector(const char* image_name, const char* label_name, 
    vector<vector<unsigned char>>& vec_image, vector<unsigned char>& vec_label, const vector<int>& res){
    int cnt = res.size();
    FILE* in_image = fopen(image_name, "rb");
    FILE* in_label = fopen(label_name, "rb");

    for (int i = 0; i < cnt; ++i){
        fseek(in_image, 16L + res[i] * IMAGE_SIZE, SEEK_SET);
        fseek(in_label, 8L + res[i], SEEK_SET);

        vector<unsigned char> cur_image(IMAGE_SIZE, 0);
        unsigned char cur_label = 0;

        for (int j = 0; j < IMAGE_SIZE; ++j)    
            fread(&cur_image[j], 1, 1, in_image);
        fread(&cur_label, 1, 1, in_label);

        vec_label.push_back(cur_label);
        vec_image.push_back(cur_image);
    }

    fclose(in_image);
    fclose(in_label);
}

void normalization(const vector<vector<unsigned char> >& vec_image, vector<vector<Mat_n> >& res){
    int r = vec_image.size(), c = vec_image[0].size();
    for (int i = 0; i < r; ++i){
        res.push_back(vector<Mat_n>());
        for (int j = 0; j < c; ++j) 
            res[i].push_back(1.0 * vec_image[i][j] / 255);
    }
}

void compare(const char* label_name, const vector<int>& pre_res, vector<int>& TP, 
    vector<int>& TN, vector<int>& FP, vector<int>& FN){
    /* 返回对应的统计数据 */
    FILE* in_label = fopen(label_name, "rb");
    fseek(in_label, 8L, SEEK_SET);
    vector<unsigned char> real_labels;

    for (int i = 0; i < TEST_SIZE; ++i){
        unsigned char cur_label = 0;
        fread(&cur_label, 1, 1, in_label);
        real_labels.push_back(cur_label);
    }
    
    for (int i = 0; i < 10; ++i){
        for (int j = 0; j < TEST_SIZE; ++j){
            if(real_labels[j] == i){
                if(pre_res[j] == i) TP[i]++;
                else FN[i]++;
            }else{
                if(pre_res[j] == i) FP[i]++;
                else TN[i]++;
            }
        }
    }
}