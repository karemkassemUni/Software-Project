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

/*******************
 * Helper Functions
 *******************/
/* Matrix creation and memory management */
matrix* create_matrix(int rows, int cols);
void free_matrix(matrix* mat);

/* Basic matrix operations */
void matrix_multiply(matrix* result, matrix* mat1, matrix* mat2, int transpose2);
double frobenius_norm_diff(matrix* mat1, matrix* mat2);
void print_matrix(matrix* mat);

/*******************
 * Core Algorithm Functions
 *******************/
/* Similarity matrix calculations */
matrix* calculate_similarity_matrix(const double* points, int n, int d);
matrix* calculate_diagonal_degree_matrix(matrix* sim_matrix);
matrix* calculate_normalized_similarity(matrix* sim_matrix, matrix* degree_matrix);

/* Matrix factorization */
matrix* optimize_h(matrix* w, matrix* h_init, int max_iter, double epsilon);

/*******************
 * Data Handling Functions
 *******************/
/* File I/O */
double* read_data(const char* filename, int* n, int* d);
double euclidean_distance_squared(const double* point1, const double* point2, int d);

#endif /* SYMNMF_H_ */