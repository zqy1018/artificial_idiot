#ifndef MY_MATRIX

    #define MY_MATRIX

    #include <vector>
    using std::vector;

    typedef double Mat_n;

    enum Matrix_constructor_mode {ALL_ZERO, ALL_ONE, IDENTITY, RANDOM};

    const Mat_n eps = 1e-7;

    int cmp(Mat_n arg);

    class Matrix{
        friend Matrix operator+ (const Matrix& obj1, const Matrix& obj2);
        friend Matrix operator+ (Mat_n obj1, const Matrix& obj2);
        friend Matrix operator+ (const Matrix& obj1, Mat_n obj2);

        friend Matrix operator- (const Matrix& obj1, const Matrix& obj2);
        friend Matrix operator- (Mat_n obj1, const Matrix& obj2);
        friend Matrix operator- (const Matrix& obj1, Mat_n obj2);

        friend Matrix operator* (const Matrix& obj1, const Matrix& obj2);
        friend Matrix operator* (Mat_n obj1, const Matrix& obj2);
        friend Matrix operator* (const Matrix& obj1, Mat_n obj2);

        friend Matrix operator/ (const Matrix& obj1, const Matrix& obj2);
        friend Matrix operator/ (Mat_n obj1, const Matrix& obj2);
        friend Matrix operator/ (const Matrix& obj1, Mat_n obj2);

        friend Matrix operator^ (const Matrix& obj1, const Matrix& obj2);
        /* 矩阵乘法 */

        friend bool operator== (const Matrix& obj1, const Matrix& obj2);
        friend bool operator!= (const Matrix& obj1, const Matrix& obj2);
    protected:
        Mat_n **mat;
        int row, column;
    public:
        explicit Matrix(int _r = 1, int _c = 1, int _mode = ALL_ZERO);
        Matrix(const Matrix& obj); 
        Matrix(const Matrix&& obj);
        Matrix(const vector<Mat_n>& obj); 

        Mat_n   sum() const;
        Matrix  T() const;
        Matrix& operator= (const Matrix& obj);
        Matrix& operator= (const Matrix&& obj);
        Mat_n* operator[] (int i);
        const Mat_n* operator[] (int i) const;
        explicit operator Mat_n() const;
        int getRow() const;
        int getColumn() const;

        bool isZero() const;
        bool isOne() const;
        bool isIdentity() const;

        ~Matrix();
    };

#endif