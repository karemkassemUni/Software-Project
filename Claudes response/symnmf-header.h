#ifndef SYMNMF_H
#define SYMNMF_H

/* Data structures */
typedef struct {
    double** data;
    int rows;
    int cols;
} Matrix;

/* Matrix operations */
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix* matrix);
Matrix* matrix_multiply(Matrix* a, Matrix* b);
Matrix* matrix_transpose(Matrix* a);
double frobenius_norm_diff(Matrix* a, Matrix* b);

/* Core algorithm functions */
Matrix* calculate_similarity_matrix(Matrix* points);
Matrix* calculate_diagonal_degree_matrix(Matrix* similarity);
Matrix* calculate_normalized_similarity(Matrix* similarity, Matrix* degree);
Matrix* perform_symnmf(Matrix* h_init, Matrix* w, int max_iter, double epsilon);

/* Utility functions */
double euclidean_distance(double* a, double* b, int dim);
Matrix* read_data_from_file(const char* filename);
void print_matrix(Matrix* matrix);

#endif /* SYMNMF_H */
