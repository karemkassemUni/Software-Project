#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"
#include <float.h>

#define MAX_LINE_LENGTH 1024
#define EPS 1e-10

/*******************
 * Helper Functions
 *******************/

/* Calculate euclidean distance squared between two points */
double euclidean_distance_squared(const double* point1, const double* point2, int d) {
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}

/* Matrix creation with proper initialization */
matrix* create_matrix(int rows, int cols) {
    matrix* mat = (matrix*)malloc(sizeof(matrix));
    if (!mat) return NULL;
    
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double*)calloc(rows * cols, sizeof(double));
    
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    
    return mat;
}

/* Safe matrix cleanup */
void free_matrix(matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

//* Updated matrix multiplication for better numerical stability */
void matrix_multiply(matrix* result, matrix* mat1, matrix* mat2, int transpose2) {
    int i, j, k;
    int m = mat1->rows;
    int p = mat1->cols;
    int n = transpose2 ? mat2->rows : mat2->cols;
    
    // Verify dimensions are compatible
    if (p != (transpose2 ? mat2->cols : mat2->rows)) {
        return;
    }
    
    // Verify result matrix has correct dimensions
    if (result->rows != m || result->cols != n) {
        return;
    }
    
    // Allocate temporary buffer
    double* temp = (double*)calloc(m * n, sizeof(double));
    if (!temp) return;
    
    // Perform multiplication with better numerical stability
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < p; k++) {
                double val1 = mat1->data[i * p + k];
                double val2;
                if (transpose2) {
                    val2 = mat2->data[j * mat2->cols + k];
                } else {
                    val2 = mat2->data[k * mat2->cols + j];
                }
                
                // Skip multiplication if either value is very small
                if (fabs(val1) < EPS || fabs(val2) < EPS) {
                    continue;
                }
                
                sum += val1 * val2;
            }
            temp[i * n + j] = sum;
        }
    }
    
    // Copy result and cleanup
    memcpy(result->data, temp, m * n * sizeof(double));
    free(temp);
}

/* Frobenius norm calculation */
double frobenius_norm_diff(matrix* mat1, matrix* mat2) {
    if (!mat1 || !mat2 || mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return INFINITY;
    }
    
    double sum = 0.0;
    int size = mat1->rows * mat1->cols;
    
    for (int i = 0; i < size; i++) {
        double diff = mat1->data[i] - mat2->data[i];
        sum += diff * diff;
    }
    
    return sqrt(sum);
}

/*******************
 * Main Algorithm Functions
 *******************/

/* Calculate similarity matrix */
matrix* calculate_similarity_matrix(const double* points, int n, int d) {
    matrix* sim = create_matrix(n, n);
    if (!sim) return NULL;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dist = euclidean_distance_squared(&points[i*d], &points[j*d], d);
                sim->data[i*n + j] = exp(-dist/2);
            }
        }
    }
    
    return sim;
}

/* Calculate diagonal degree matrix */
matrix* calculate_diagonal_degree_matrix(matrix* sim_matrix) {
    int n = sim_matrix->rows;
    matrix* degree = create_matrix(n, n);
    if (!degree) return NULL;
    
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += sim_matrix->data[i * n + j];
        }
        degree->data[i * n + i] = sum;
    }
    
    return degree;
}

/* Calculate normalized similarity matrix */
matrix* calculate_normalized_similarity(matrix* sim_matrix, matrix* degree_matrix) {
    int n = sim_matrix->rows;
    matrix* norm = create_matrix(n, n);
    if (!norm) return NULL;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d_ii = sqrt(degree_matrix->data[i*n + i]);
            double d_jj = sqrt(degree_matrix->data[j*n + j]);
            
            if (d_ii > EPS && d_jj > EPS) {
                norm->data[i*n + j] = sim_matrix->data[i*n + j] / (d_ii * d_jj);
            }
        }
    }
    
    return norm;
}

/* Matrix factorization optimization */
matrix* optimize_h(matrix* w, matrix* h_init, int max_iter, double epsilon) {
    int n = h_init->rows;
    int k = h_init->cols;
    double beta = BETA;
    double diff;
    int iter;
    
    // Create all needed matrices
    matrix* h_curr = create_matrix(n, k);
    matrix* h_prev = create_matrix(n, k);
    matrix* wh = create_matrix(n, k);
    matrix* hth = create_matrix(k, k);
    matrix* h_htH = create_matrix(n, k);
    matrix* result = NULL;
    
    // Check all allocations succeeded
    if (!h_curr || !h_prev || !wh || !hth || !h_htH) {
        goto cleanup;
    }
    
    // Initialize H with h_init
    memcpy(h_curr->data, h_init->data, n * k * sizeof(double));
    
    // Main optimization loop
    for (iter = 0; iter < max_iter; iter++) {
        // Store current H
        memcpy(h_prev->data, h_curr->data, n * k * sizeof(double));
        
        // Calculate WH
        matrix_multiply(wh, w, h_curr, 0);
        
        // Calculate H^T H (k x k matrix)
        matrix_multiply(hth, h_curr, h_curr, 1);
        
        // Calculate H(H^T H)
        matrix_multiply(h_htH, h_curr, hth, 0);
        
        // Update H
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double numerator = wh->data[i * k + j];
                double denominator = h_htH->data[i * k + j];
                
                if (denominator > 1e-10) {
                    h_curr->data[i * k + j] = h_curr->data[i * k + j] * 
                        ((1.0 - beta) + beta * (numerator / denominator));
                    
                    // Ensure value stays positive and finite
                    if (h_curr->data[i * k + j] < 1e-10 || !isfinite(h_curr->data[i * k + j])) {
                        h_curr->data[i * k + j] = 1e-10;
                    }
                }
            }
        }
        
        // Check convergence using relative change
        diff = 0.0;
        for (int i = 0; i < n * k; i++) {
            double rel_diff = fabs(h_curr->data[i] - h_prev->data[i]);
            if (h_prev->data[i] > 1e-10) {
                rel_diff /= h_prev->data[i];
            }
            diff = fmax(diff, rel_diff);
        }
        
        if (diff < epsilon) {
            break;
        }
    }
    
    // Create and fill result matrix
    result = create_matrix(n, k);
    if (result) {
        memcpy(result->data, h_curr->data, n * k * sizeof(double));
        
        // Normalize columns to sum to 1
        for (int j = 0; j < k; j++) {
            double col_sum = 0.0;
            for (int i = 0; i < n; i++) {
                col_sum += result->data[i * k + j];
            }
            if (col_sum > 1e-10) {
                for (int i = 0; i < n; i++) {
                    result->data[i * k + j] /= col_sum;
                }
            }
        }
    }
    
cleanup:
    if (h_curr) free_matrix(h_curr);
    if (h_prev) free_matrix(h_prev);
    if (wh) free_matrix(wh);
    if (hth) free_matrix(hth);
    if (h_htH) free_matrix(h_htH);
    
    return result;
}

/* Read data from file */
double* read_data(const char* filename, int* n, int* d) {
    FILE* fp;
    char line[MAX_LINE_LENGTH];
    char* token;
    double* points = NULL;
    int row = 0, col;
    
    *n = 0;
    *d = 0;
    
    fp = fopen(filename, "r");
    if (!fp) {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    
    /* Count dimensions from first line */
    if (fgets(line, MAX_LINE_LENGTH, fp)) {
        token = strtok(line, ",");
        while (token) {
            (*d)++;
            token = strtok(NULL, ",");
        }
    }
    
    /* Count number of points */
    rewind(fp);
    while (fgets(line, MAX_LINE_LENGTH, fp)) {
        (*n)++;
    }
    
    points = (double*)malloc((*n) * (*d) * sizeof(double));
    if (!points) {
        fclose(fp);
        printf("An Error Has Occurred\n");
        return NULL;
    }
    
    rewind(fp);
    row = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp) && row < *n) {
        col = 0;
        token = strtok(line, ",");
        while (token && col < *d) {
            points[row * (*d) + col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        if (col != *d) {
            free(points);
            fclose(fp);
            printf("An Error Has Occurred\n");
            return NULL;
        }
        row++;
    }
    
    fclose(fp);
    return points;
}

/* Print matrix */
void print_matrix(matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%.4f", mat->data[i * mat->cols + j]);
            if (j < mat->cols - 1) printf(",");
        }
        printf("\n");
    }
}

/* Main function */
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    char* goal = argv[1];
    char* filename = argv[2];
    int n, d;
    
    double* points = read_data(filename, &n, &d);
    if (!points) return 1;
    
    matrix *sim = NULL, *ddg = NULL, *norm = NULL;
    
    if (strcmp(goal, "sym") == 0) {
        sim = calculate_similarity_matrix(points, n, d);
        if (!sim) {
            free(points);
            printf("An Error Has Occurred\n");
            return 1;
        }
        print_matrix(sim);
        free_matrix(sim);
    }
    else if (strcmp(goal, "ddg") == 0) {
        sim = calculate_similarity_matrix(points, n, d);
        if (!sim) {
            free(points);
            printf("An Error Has Occurred\n");
            return 1;
        }
        ddg = calculate_diagonal_degree_matrix(sim);
        if (!ddg) {
            free_matrix(sim);
            free(points);
            printf("An Error Has Occurred\n");
            return 1;
        }
        print_matrix(ddg);
        free_matrix(sim);
        free_matrix(ddg);
    }
    else if (strcmp(goal, "norm") == 0) {
        sim = calculate_similarity_matrix(points, n, d);
        if (!sim) {
            free(points);
            printf("An Error Has Occurred\n");
            return 1;
        }
        ddg = calculate_diagonal_degree_matrix(sim);
        if (!ddg) {
            free_matrix(sim);
            free(points);
            printf("An Error Has Occurred\n");
            return 1;
        }
        norm = calculate_normalized_similarity(sim, ddg);
        if (!norm) {
            free_matrix(sim);
            free_matrix(ddg);
            free(points);
            printf("An Error Has Occurred\n");
            return 1;
        }
        print_matrix(norm);
        free_matrix(sim);
        free_matrix(ddg);
        free_matrix(norm);
    }
    else {
        free(points);
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    free(points);
    return 0;
}