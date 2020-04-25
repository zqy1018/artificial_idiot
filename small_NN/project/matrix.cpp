#include "matrix.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>

int cmp(Mat_n arg){
    if(fabs(arg) < eps)
        return 0;
    return arg > 0 ? 1: -1;
}

Matrix::Matrix(int _r, int _c, int _mode){
    row = _r, column = _c;
    mat = new Mat_n*[_r];
    if(_mode == ALL_ZERO){
        for (int i = 0; i < row; ++i)
            mat[i] = new Mat_n[_c]{0};
    }else if(_mode == ALL_ONE){
        for (int i = 0; i < row; ++i)
            mat[i] = new Mat_n[_c]{1};
    }else if(_mode == IDENTITY){
        /* 必须满足row == column */
        assert(row == column);
        for (int i = 0; i < row; ++i){
            mat[i] = new Mat_n[_c]{0};
            mat[i][i] = 1;
        }   
    }else if(_mode == RANDOM){
        /* [0, 1) 随机矩阵 */
        srand(time(NULL));
        for (int i = 0; i < row; ++i){
            mat[i] = new Mat_n[_c]{0};
            for (int j = 0; j < column; ++j)
                mat[i][j] = (Mat_n)rand() / RAND_MAX, 
                mat[i][j] -= 0.5;
        }   
    }
}

Matrix::Matrix(const vector<Mat_n>& obj){
    row = obj.size(), column = 1; // 列向量
    mat = new Mat_n*[row];
    for (int i = 0; i < row; ++i){
        mat[i] = new Mat_n[column];
        mat[i][0] = obj[i];
    }
}

Matrix::Matrix(const Matrix& obj){
    row = obj.row, column = obj.column;
    mat = new Mat_n*[row];
    for (int i = 0; i < row; ++i){
        mat[i] = new Mat_n[column];
        for (int j = 0; j < column; ++j)
            mat[i][j] = obj.mat[i][j];
    }
}

Matrix::Matrix(const Matrix&& obj){
    /* 移动构造函数 */
    row = obj.row, column = obj.column;
    mat = new Mat_n*[row];
    for (int i = 0; i < row; ++i){
        mat[i] = new Mat_n[column];
        for (int j = 0; j < column; ++j)
            mat[i][j] = obj.mat[i][j];
    }
}

Matrix& Matrix::operator= (const Matrix& obj){
    if(&obj == this) return *this;
    for (int i = 0; i < this->row; ++i)
        delete [] mat[i];
    delete [] mat;
    row = obj.row, column = obj.column;
    mat = new Mat_n*[row];
    for (int i = 0; i < row; ++i){
        mat[i] = new Mat_n[column];
        for (int j = 0; j < column; ++j)
            mat[i][j] = obj.mat[i][j];
    }
    return *this;
}

Matrix& Matrix::operator= (const Matrix&& obj){
    /* 移动赋值函数 */
    for (int i = 0; i < this->row; ++i)
        delete [] mat[i];
    delete [] mat;
    row = obj.row, column = obj.column;
    mat = new Mat_n*[row];
    for (int i = 0; i < row; ++i){
        mat[i] = new Mat_n[column];
        for (int j = 0; j < column; ++j)
            mat[i][j] = obj.mat[i][j];
    }
    return *this;
}

Matrix::~Matrix(){
    for (int i = 0; i < row; ++i)
        delete [] mat[i];
    delete [] mat;
}

Matrix operator+ (const Matrix& obj1, const Matrix& obj2){
    /* 禁止广播功能 */
    assert(obj1.row == obj2.row && obj1.column == obj2.column);
    Matrix res(obj1.row, obj1.column, ALL_ZERO);
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            res[i][j] = obj1.mat[i][j] + obj2.mat[i][j];
    return res;
}

Matrix operator+ (Mat_n obj1, const Matrix& obj2){
    Matrix res(obj2.row, obj2.column, ALL_ZERO);
    for (int i = 0; i < obj2.row; ++i)
        for (int j = 0; j < obj2.column; ++j)
            res[i][j] = obj2.mat[i][j] + obj1;
    return res;
}

Matrix operator+ (const Matrix& obj1, Mat_n obj2){
    Matrix res(obj1.row, obj1.column, ALL_ZERO);
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            res[i][j] = obj1.mat[i][j] + obj2;
    return res;
}

Matrix operator- (const Matrix& obj1, const Matrix& obj2){
    /* 禁止广播功能 */
    assert(obj1.row == obj2.row && obj1.column == obj2.column);
    Matrix res(obj1.row, obj1.column, ALL_ZERO);
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            res[i][j] = obj1.mat[i][j] - obj2.mat[i][j];
    return res;
}

Matrix operator- (Mat_n obj1, const Matrix& obj2){
    Matrix res(obj2.row, obj2.column, ALL_ZERO);
    for (int i = 0; i < obj2.row; ++i)
        for (int j = 0; j < obj2.column; ++j)
            res[i][j] = obj1 - obj2.mat[i][j];
    return res;
}

Matrix operator- (const Matrix& obj1, Mat_n obj2){
    Matrix res(obj1.row, obj1.column, ALL_ZERO);
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            res[i][j] = obj1.mat[i][j] - obj2;
    return res;
}

Matrix operator* (const Matrix& obj1, const Matrix& obj2){
    /* 禁止广播功能 */
    assert(obj1.row == obj2.row && obj1.column == obj2.column);
    Matrix res(obj1.row, obj1.column, ALL_ZERO);
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            res[i][j] = obj1.mat[i][j] * obj2.mat[i][j];
    return res;
}

Matrix operator* (Mat_n obj1, const Matrix& obj2){
    Matrix res(obj2.row, obj2.column, ALL_ZERO);
    for (int i = 0; i < obj2.row; ++i)
        for (int j = 0; j < obj2.column; ++j)
            res[i][j] = obj2.mat[i][j] * obj1;
    return res;
}

Matrix operator* (const Matrix& obj1, Mat_n obj2){
    Matrix res(obj1.row, obj1.column, ALL_ZERO);
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            res[i][j] = obj1.mat[i][j] * obj2;
    return res;
}

Matrix operator/ (const Matrix& obj1, const Matrix& obj2){
    /* 禁止广播功能 */
    assert(obj1.row == obj2.row && obj1.column == obj2.column);
    Matrix res(obj1.row, obj1.column, ALL_ZERO);
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            res[i][j] = obj1.mat[i][j] / obj2.mat[i][j];
    return res;
}

Matrix operator/ (Mat_n obj1, const Matrix& obj2){
    Matrix res(obj2.row, obj2.column, ALL_ZERO);
    for (int i = 0; i < obj2.row; ++i)
        for (int j = 0; j < obj2.column; ++j)
            res[i][j] = obj1 / obj2.mat[i][j];
    return res;
}

Matrix operator/ (const Matrix& obj1, Mat_n obj2){
    Matrix res(obj1.row, obj1.column, ALL_ZERO);
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            res[i][j] = obj1.mat[i][j] / obj2;
    return res;
}

Matrix operator^ (const Matrix& obj1, const Matrix& obj2){
    assert(obj1.column == obj2.row);
    Matrix res(obj1.row, obj2.column);
    for (int i = 0; i < obj1.row; ++i)
        for (int k = 0; k < obj2.column; ++k)
            for (int j = 0; j < obj1.column; ++j)
                res.mat[i][k] += obj1.mat[i][j] * obj2.mat[j][k];
    return res;
}

bool operator== (const Matrix& obj1, const Matrix& obj2){
    if(obj1.row != obj2.row || obj1.column != obj2.column)  
        return false;
    for (int i = 0; i < obj1.row; ++i)
        for (int j = 0; j < obj1.column; ++j)
            if(cmp(obj1.mat[i][j] - obj2.mat[i][j]) != 0)
                return false;
    return true;
}

bool operator!= (const Matrix& obj1, const Matrix& obj2){
    return !(obj1 == obj2);
}

Mat_n Matrix::sum() const{
    Mat_n res = 0;
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < column; ++j)
            res += mat[i][j];
    return res;
}

Matrix Matrix::T() const{
    /* 转置 */
    Matrix res(column, row, ALL_ZERO);
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < column; ++j)
            res[j][i] = mat[i][j];
    return res;
}

Matrix::operator Mat_n() const{
    assert(row == 1 && column == 1);
    return mat[0][0];
}

bool Matrix::isZero() const{
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < column; ++j)
            if(cmp(mat[i][j]) != 0) 
                return false;
    return true;
}

bool Matrix::isOne() const{
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < column; ++j)
            if(cmp(mat[i][j] - 1.0) != 0) 
                return false;
    return true;
}

bool Matrix::isIdentity() const{
    if(row != column) return false;
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < column; ++j){
            if(i == j){
                if(cmp(mat[i][j]) != 0) 
                    return false;
            }else{
                if(cmp(mat[i][j] - 1.0) != 0) 
                    return false;
            }
        }
    return true;
}

int Matrix::getRow() const{
    return row;
}

int Matrix::getColumn() const{
    return column;
}

Mat_n* Matrix::operator[] (int i){
    return mat[i];
}

const Mat_n* Matrix::operator[] (int i) const{
    return mat[i];
}
