#ifndef SYMNMF_H_
#define SYMNMF_H_

/* Constants */
#define MAX_ITER 300
#define EPSILON 0.0001
#define BETA 0.5

/* Structure to hold matrix data */
typedef struct {
    int rows;
    int cols;
    double* data;
} matrix;

/* Function declarations */
matrix* create_matrix(int rows, int cols);
void free_matrix(matrix* mat);
matrix* calculate_similarity(const double* points, int n, int d);
matrix* calculate_degree(const matrix* sim);
matrix* calculate_normalized(const matrix* sim, const matrix* deg);
matrix* optimize_h(const matrix* w, const matrix* h_init, int n, int k);
double squared_euclidean_distance(const double* p1, const double* p2, int dim);
void matrix_multiply(const matrix* mat1, const matrix* mat2, matrix* result, int transpose2);
double frobenius_norm(const matrix* mat1, const matrix* mat2);
static void print_matrix(const matrix* mat);

#endif /* SYMNMF_H_ */