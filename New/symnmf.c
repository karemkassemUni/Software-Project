#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define MAX_LINE_LENGTH 1024

/* Helper function to safely copy matrix data */
static void copy_matrix_data(const matrix* src, matrix* dst) {
    int size = src->rows * src->cols;
    for (int i = 0; i < size; i++) {
        dst->data[i] = src->data[i];
    }
}

/* Matrix creation and memory management */
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

void free_matrix(matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

/* Matrix operations */
double squared_euclidean_distance(const double* p1, const double* p2, int dim) {
    double sum = 0.0;
    int i;
    for (i = 0; i < dim; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

void matrix_multiply(const matrix* mat1, const matrix* mat2, matrix* result, int transpose2) {
    int i, j, k;
    double sum, val1, val2;
    
    /* Validate input */
    if (!mat1 || !mat2 || !result) return;
    
    int m = mat1->rows;
    int p = mat1->cols;
    int n = transpose2 ? mat2->rows : mat2->cols;
    /* Validate dimensions */
    if (p != (transpose2 ? mat2->cols : mat2->rows) || result->rows != m || result->cols != n) {
        printf("Invalid dimensions\n");
        return;
    }
    
    /* Clear result matrix */
    for (i = 0; i < m * n; i++) {
        result->data[i] = 0.0;
    }
    
    /* Perform multiplication */
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < p; k++) {
                val1 = mat1->data[i * p + k];
                val2 = transpose2 ? mat2->data[j * mat2->cols + k] : mat2->data[k * mat2->cols + j];
                
                sum += val1 * val2;
            }
            result->data[i * n + j] = sum;
        }
    }
}

double frobenius_norm(const matrix* mat1, const matrix* mat2) {
    if (!mat1 || !mat2 || 
        mat1->rows != mat2->rows || 
        mat1->cols != mat2->cols) {
        return INFINITY;
    }
    
    double sum = 0.0;
    int size = mat1->rows * mat1->cols;
    
    for (int i = 0; i < size; i++) {
        double diff = mat1->data[i] - mat2->data[i];
        sum += diff * diff;
    }
    
    return sum;
}

/* Core algorithm functions */
matrix* calculate_similarity(const double* points, int n, int d) {
    matrix* sim = create_matrix(n, n);
    int i, j;
    
    if (!sim) return NULL;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                double dist = squared_euclidean_distance(&points[i * d], &points[j * d], d);
                sim->data[i * n + j] = exp(-dist/2);
            }
        }
    }
    
    return sim;
}

matrix* calculate_degree(const matrix* sim) {
    matrix* deg = create_matrix(sim->rows, sim->rows);
    if (!deg) return NULL;
    
    for (int i = 0; i < sim->rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < sim->cols; j++) {
            sum += sim->data[i * sim->cols + j];
        }
        deg->data[i * deg->cols + i] = sum;
    }
    
    return deg;
}

matrix* calculate_normalized(const matrix* sim, const matrix* deg) {
    matrix* norm = create_matrix(sim->rows, sim->cols);
    if (!norm) return NULL;
    
    for (int i = 0; i < sim->rows; i++) {
        for (int j = 0; j < sim->cols; j++) {
            double d_ii = deg->data[i * deg->cols + i];
            double d_jj = deg->data[j * deg->cols + j];
            
            if (d_ii > 0.0 && d_jj > 0.0) {
                d_ii = sqrt(d_ii);
                d_jj = sqrt(d_jj);
                norm->data[i * norm->cols + j] = sim->data[i * sim->cols + j] / (d_ii * d_jj);
            }
        }
    }
    
    return norm;
}

matrix* optimize_h(const matrix* w, const matrix* h_init, int n, int k) {
    matrix *h_curr = NULL, *h_next = NULL, *result = NULL;
    matrix *wh = NULL, *hht = NULL, *hhth = NULL;
    double diff = INFINITY;
    int iter;
    
    // Allocate matrices
    h_curr = create_matrix(n, k);
    h_next = create_matrix(n, k);
    wh = create_matrix(n, k);
    hht = create_matrix(n, n);
    hhth = create_matrix(n, k);
    
    if (!h_curr || !h_next || !wh || !hht || !hhth) {
        printf("An Error Has Occurred\n"); // delete this later
        goto cleanup;
    }
    
    // Initialize
    copy_matrix_data(h_init, h_curr);
    

    // Main iteration loop
    for (iter = 0; iter < MAX_ITER && diff >= EPSILON; iter++) {
        // Zero out h_next
        memset(h_next->data, 0, n * k * sizeof(double));

        // Calculate WH
        matrix_multiply(w, h_curr, wh, 0);
        
        // Calculate HH^T
        matrix_multiply(h_curr, h_curr, hht, 1);
        
        // Calculate (HH^T)H
        matrix_multiply(hht, h_curr, hhth, 0);
        
        // Update rule
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double wh_ij = wh->data[i * k + j];
                double hhth_ij = hhth->data[i * k + j];
                double h_ij = h_curr->data[i * k + j];
                
                // if (h_ij < 1e-10) {
                //     h_next->data[i * k + j] = 0.0;
                //     continue;
                // }
                
                double update;
                update = (1.0 - BETA) + BETA * (wh_ij / hhth_ij);
                // if (hhth_ij > 1e-10) {
                //     update = (1.0 - BETA) + BETA * (wh_ij / hhth_ij);
                // } else {
                //     update = 1.0 - BETA;
                // }
                
                h_next->data[i * k + j] = h_ij * update;
                
                // Handle very small values
                // if (h_next->data[i * k + j] < 1e-10) {
                //     h_next->data[i * k + j] = 0.0;
                // }
            }
        }

        // Calculate difference
        diff = frobenius_norm(h_next, h_curr);
        
        // Copy h_next to h_curr
        copy_matrix_data(h_next, h_curr);
    }
    
    // Create result
    result = create_matrix(n, k);
    if (result) {
        copy_matrix_data(h_curr, result);
    }
    
cleanup:
    free_matrix(h_curr);
    free_matrix(h_next);
    free_matrix(wh);
    free_matrix(hht);
    free_matrix(hhth);
    
    return result;
}


/* File reading and main function */
static double* read_data_points(const char* filename, int* n_points, int* n_dim) {
    FILE* fp;
    char line[MAX_LINE_LENGTH];
    char* token;
    double* points = NULL;
    int row = 0, col;
    
    *n_points = 0;
    *n_dim = 0;
    
    fp = fopen(filename, "r");
    if (!fp) {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    
    /* Count dimensions from first line */
    if (fgets(line, MAX_LINE_LENGTH, fp)) {
        token = strtok(line, ",");
        while (token) {
            (*n_dim)++;
            token = strtok(NULL, ",");
        }
    }
    
    /* Count number of points */
    rewind(fp);
    while (fgets(line, MAX_LINE_LENGTH, fp)) {
        (*n_points)++;
    }
    
    points = (double*)malloc((*n_points) * (*n_dim) * sizeof(double));
    if (!points) {
        fclose(fp);
        printf("An Error Has Occurred\n");
        return NULL;
    }
    
    rewind(fp);
    while (fgets(line, MAX_LINE_LENGTH, fp) && row < *n_points) {
        col = 0;
        token = strtok(line, ",");
        while (token && col < *n_dim) {
            points[row * (*n_dim) + col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        if (col != *n_dim) {
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

static void print_matrix(const matrix* mat) {
    int i, j;
    
    for (i = 0; i < mat->rows; i++) {
        for (j = 0; j < mat->cols; j++) {
            printf("%.4f", mat->data[i * mat->cols + j]);
            if (j < mat->cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    const char* goal = argv[1];
    const char* filename = argv[2];
    int n, d;
    double* points;
    matrix *sim = NULL, *deg = NULL, *norm = NULL;
    
    points = read_data_points(filename, &n, &d);
    if (!points) {
        return 1;
    }
    
    if (strcmp(goal, "sym") == 0) {
        sim = calculate_similarity(points, n, d);
        if (!sim) {
            free(points);
            return 1;
        }
        print_matrix(sim);
        free_matrix(sim);
    }
    else if (strcmp(goal, "ddg") == 0) {
        sim = calculate_similarity(points, n, d);
        if (!sim) {
            free(points);
            return 1;
        }
        
        deg = calculate_degree(sim);
        if (!deg) {
            free_matrix(sim);
            free(points);
            return 1;
        }
        
        print_matrix(deg);
        free_matrix(sim);
        free_matrix(deg);
    }
    else if (strcmp(goal, "norm") == 0) {
        sim = calculate_similarity(points, n, d);
        if (!sim) {
            free(points);
            return 1;
        }
        
        deg = calculate_degree(sim);
        if (!deg) {
            free_matrix(sim);
            free(points);
            return 1;
        }
        
        norm = calculate_normalized(sim, deg);
        if (!norm) {
            free_matrix(sim);
            free_matrix(deg);
            free(points);
            return 1;
        }
        
        print_matrix(norm);
        free_matrix(sim);
        free_matrix(deg);
        free_matrix(norm);
    }
    else {
        printf("An Error Has Occurred\n");
        free(points);
        return 1;
    }
    
    free(points);
    return 0;
}