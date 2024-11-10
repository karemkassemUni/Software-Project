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
                
                /* Skip multiplication if either value is very small */
                if (fabs(val1) < EPS || fabs(val2) < EPS) {
                    continue;
                }
                
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
    
    return sqrt(sum);
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
    int i, j;
    
    if (!deg) return NULL;
    
    for (i = 0; i < sim->rows; i++) {
        double sum = 0.0;
        for (j = 0; j < sim->cols; j++) {
            sum += sim->data[i * sim->cols + j];
        }
        deg->data[i * deg->cols + i] = sum;
    }
    
    return deg;
}

matrix* calculate_normalized(const matrix* sim, const matrix* deg) {
    matrix* norm = create_matrix(sim->rows, sim->cols);
    int i, j;
    
    if (!norm) return NULL;
    
    for (i = 0; i < sim->rows; i++) {
        for (j = 0; j < sim->cols; j++) {
            double d_ii = sqrt(deg->data[i * deg->cols + i]);
            double d_jj = sqrt(deg->data[j * deg->cols + j]);
            
            if (d_ii > EPS && d_jj > EPS) {
                norm->data[i * norm->cols + j] = sim->data[i * sim->cols + j] / (d_ii * d_jj);
            }
        }
    }
    
    return norm;
}

matrix* optimize_h(const matrix* w, const matrix* h_init, int n, int k) {
    matrix* h_curr = NULL;
    matrix* h_prev = NULL;
    matrix* wh = NULL;
    matrix* hth = NULL;
    matrix* h_hth = NULL;
    matrix* result = NULL;
    double diff;
    int iter;
    
    /* Validate input */
    if (!w || !h_init || n <= 0 || k <= 0 || 
        w->rows != n || w->cols != n || 
        h_init->rows != n || h_init->cols != k) {
        return NULL;
    }
    
    /* Create all needed matrices */
    h_curr = create_matrix(n, k);
    h_prev = create_matrix(n, k);
    wh = create_matrix(n, k);
    hth = create_matrix(n, n);
    h_hth = create_matrix(n, k);
    
    if (!h_curr || !h_prev || !wh || !hth || !h_hth) {
        goto cleanup;
    }
    
    /* Initialize h_curr with h_init */
    copy_matrix_data(h_init, h_curr);
    
    /* Main optimization loop */
    for (iter = 0; iter < MAX_ITER; iter++) {
        /* Store current H */
        copy_matrix_data(h_curr, h_prev);
        
        /* Calculate WH */
        matrix_multiply(w, h_curr, wh, 0);
        
        /* Calculate H H^T */
        matrix_multiply(h_curr, h_curr, hth, 1);
        
        /* Calculate (H^T H)H  [{nxn}{nxk}+>{nxk}]*/
        matrix_multiply(hth, h_curr, h_hth, 0);
        
        /* Update H */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double current_value = h_curr->data[i * k + j];
                double wh_value = wh->data[i * k + j];
                double h_hth_value = h_hth->data[i * k + j];
                
                if (h_hth_value > EPS) {
                    double update = (1.0 - BETA) + BETA * (wh_value / h_hth_value);
                    h_curr->data[i * k + j] = current_value * update;
                }
                
                /* Ensure value stays positive */
                if (h_curr->data[i * k + j] < EPS) {
                    h_curr->data[i * k + j] = EPS;
                }
            }
        }
        
        /* Check convergence */
        diff = frobenius_norm(h_curr, h_prev);
        if (diff < EPSILON) {
            break;
        }
    }
    
    /* Prepare result */
    result = create_matrix(n, k);
    if (result) {
        copy_matrix_data(h_curr, result);
    }
    
cleanup:
    free_matrix(h_curr);
    free_matrix(h_prev);
    free_matrix(wh);
    free_matrix(hth);
    free_matrix(h_hth);
    
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