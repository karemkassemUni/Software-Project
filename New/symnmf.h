#ifndef SYMNMF_H_
#define SYMNMF_H_

/* Constants */
#define MAX_ITER 300
#define EPSILON 1e-4
#define BETA 0.5

/* Structure to hold matrix data */
typedef struct {
    int rows;
    int cols;
    double* data;
} matrix;

/* Core functionality */
matrix* create_matrix(int rows, int cols);
void free_matrix(matrix* mat);
matrix* calculate_similarity_matrix(double* points, int n, int d);
matrix* calculate_diagonal_degree_matrix(matrix* sim_matrix);
matrix* calculate_normalized_similarity(matrix* sim_matrix, matrix* degree_matrix);
matrix* optimize_h(matrix* w, matrix* h_init, int max_iter, double epsilon);

/* Helper functions */
double euclidean_distance_squared(double* point1, double* point2, int d);
void matrix_multiply(matrix* result, matrix* mat1, matrix* mat2, int transpose2);
double frobenius_norm_diff(matrix* mat1, matrix* mat2);
matrix* matrix_transpose(matrix* mat);

#endif /* SYMNMF_H_ */