#ifndef SYMNMF_H_
#define SYMNMF_H_

/* Constants */
#define MAX_ITER 300
#define EPSILON 0.0001
#define BETA 0.5
#define EPS 1e-10

/* Structure to hold matrix data */
typedef struct {
    int rows;
    int cols;
    double* data;
} matrix;

/* Matrix creation and memory management */
matrix* create_matrix(int rows, int cols);
void free_matrix(matrix* mat);

/* Core algorithm functions */
matrix* calculate_similarity(const double* points, int n, int d);
matrix* calculate_degree(const matrix* sim);
matrix* calculate_normalized(const matrix* sim, const matrix* deg);
matrix* optimize_h(const matrix* w, const matrix* h_init, int n, int k);

/* Helper functions */
double squared_euclidean_distance(const double* p1, const double* p2, int dim);
void matrix_multiply(const matrix* mat1, const matrix* mat2, matrix* result, int transpose2);
double frobenius_norm(const matrix* mat1, const matrix* mat2);

#endif /* SYMNMF_H_ */