#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

#define EPSILON 1e-4
#define MAX_ITER 300
#define BETA 0.5

Matrix* create_matrix(int rows, int cols) {
    int i;
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix == NULL) {
        return NULL;
    }
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (double**)malloc(rows * sizeof(double*));
    if (matrix->data == NULL) {
        free(matrix);
        return NULL;
    }
    
    for (i = 0; i < rows; i++) {
        matrix->data[i] = (double*)calloc(cols, sizeof(double));
        if (matrix->data[i] == NULL) {
            while (--i >= 0) {
                free(matrix->data[i]);
            }
            free(matrix->data);
            free(matrix);
            return NULL;
        }
    }
    
    return matrix;
}

void free_matrix(Matrix* matrix) {
    int i;
    if (matrix == NULL) return;
    
    for (i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

double euclidean_distance(double* a, double* b, int dim) {
    double sum = 0.0;
    int i;
    for (i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;  /* Note: We don't take sqrt as we need squared distance */
}

Matrix* calculate_similarity_matrix(Matrix* points) {
    int i, j;
    Matrix* similarity = create_matrix(points->rows, points->rows);
    if (similarity == NULL) return NULL;
    
    for (i = 0; i < points->rows; i++) {
        for (j = 0; j < points->rows; j++) {
            if (i != j) {
                similarity->data[i][j] = exp(-euclidean_distance(
                    points->data[i], points->data[j], points->cols));
            }
            /* diagonal elements are 0 by default due to calloc */
        }
    }
    
    return similarity;
}

Matrix* calculate_diagonal_degree_matrix(Matrix* similarity) {
    int i, j;
    Matrix* degree = create_matrix(similarity->rows, similarity->rows);
    if (degree == NULL) return NULL;
    
    /* Calculate row sums and place on diagonal */
    for (i = 0; i < similarity->rows; i++) {
        for (j = 0; j < similarity->rows; j++) {
            degree->data[i][i] += similarity->data[i][j];
        }
    }
    
    return degree;
}

Matrix* calculate_normalized_similarity(Matrix* similarity, Matrix* degree) {
    int i, j;
    Matrix* normalized = create_matrix(similarity->rows, similarity->rows);
    if (normalized == NULL) return NULL;
    
    /* W = D^(-1/2) A D^(-1/2) */
    for (i = 0; i < similarity->rows; i++) {
        for (j = 0; j < similarity->rows; j++) {
            normalized->data[i][j] = similarity->data[i][j] / 
                sqrt(degree->data[i][i] * degree->data[j][j]);
        }
    }
    
    return normalized;
}

Matrix* matrix_multiply(Matrix* a, Matrix* b) {
    int i, j, k;
    Matrix* result = create_matrix(a->rows, b->cols);
    if (result == NULL) return NULL;
    
    for (i = 0; i < a->rows; i++) {
        for (j = 0; j < b->cols; j++) {
            for (k = 0; k < a->cols; k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    
    return result;
}

Matrix* matrix_transpose(Matrix* a) {
    int i, j;
    Matrix* result = create_matrix(a->cols, a->rows);
    if (result == NULL) return NULL;
    
    for (i = 0; i < a->rows; i++) {
        for (j = 0; j < a->cols; j++) {
            result->data[j][i] = a->data[i][j];
        }
    }
    
    return result;
}

double frobenius_norm_diff(Matrix* a, Matrix* b) {
    int i, j;
    double sum = 0.0;
    
    for (i = 0; i < a->rows; i++) {
        for (j = 0; j < a->cols; j++) {
            double diff = a->data[i][j] - b->data[i][j];
            sum += diff * diff;
        }
    }
    
    return sqrt(sum);
}

Matrix* perform_symnmf(Matrix* h_init, Matrix* w, int max_iter, double epsilon) {
    int iter, i, j;
    Matrix *h_prev, *h_curr, *h_next;
    Matrix *ht, *hth, *wh;
    double diff;
    
    h_curr = h_init;
    
    for (iter = 0; iter < max_iter; iter++) {
        h_prev = h_curr;
        
        /* Calculate necessary matrix products */
        ht = matrix_transpose(h_prev);
        hth = matrix_multiply(h_prev, ht);
        wh = matrix_multiply(w, h_prev);
        
        /* Create new H matrix */
        h_next = create_matrix(h_prev->rows, h_prev->cols);
        
        /* Update rule */
        for (i = 0; i < h_prev->rows; i++) {
            for (j = 0; j < h_prev->cols; j++) {
                h_next->data[i][j] = h_prev->data[i][j] * 
                    (1 - BETA + BETA * wh->data[i][j] / hth->data[i][j]);
            }
        }
        
        /* Check convergence */
        diff = frobenius_norm_diff(h_next, h_prev);
        
        /* Clean up intermediate matrices */
        if (iter > 0) free_matrix(h_curr);
        free_matrix(ht);
        free_matrix(hth);
        free_matrix(wh);
        
        h_curr = h_next;
        
        if (diff < epsilon) break;
    }
    
    return h_curr;
}

void print_matrix(Matrix* matrix) {
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->cols; j++) {
            printf("%.4f", matrix->data[i][j]);
            if (j < matrix->cols - 1) printf(",");
        }
        printf("\n");
    }
}

Matrix* read_data_from_file(const char* filename) {
    FILE* file;
    char line[1024];
    int rows = 0, cols = 0;
    Matrix* matrix;
    char* token;
    int i = 0, j;
    
    /* First pass: count rows and validate columns */
    file = fopen(filename, "r");
    if (file == NULL) return NULL;
    
    while (fgets(line, sizeof(line), file)) {
        if (rows == 0) {
            /* Count columns in first row */
            token = strtok(line, ",");
            while (token != NULL) {
                cols++;
                token = strtok(NULL, ",");
            }
        }
        rows++;
    }
    
    /* Allocate matrix */
    matrix = create_matrix(rows, cols);
    if (matrix == NULL) {
        fclose(file);
        return NULL;
    }
    
    /* Second pass: read data */
    rewind(file);
    while (fgets(line, sizeof(line), file)) {
        token = strtok(line, ",");
        j = 0;
        while (token != NULL && j < cols) {
            matrix->data[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    
    fclose(file);
    return matrix;
}
