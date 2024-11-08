#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

#define MAX_LINE_LENGTH 1024

matrix* create_matrix(int rows, int cols) {
    matrix* mat;
    mat = (matrix*)malloc(sizeof(matrix));
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
        if (mat->data) {
            free(mat->data);
        }
        free(mat);
    }
}

double euclidean_distance_squared(double* point1, double* point2, int d) {
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}

matrix* calculate_similarity_matrix(double* points, int n, int d) {
    int i, j;
    matrix* sim;
    
    sim = create_matrix(n, n);
    if (!sim) return NULL;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                /* Calculate squared Euclidean distance */
                double dist = euclidean_distance_squared(&points[i*d], &points[j*d], d);
                /* Apply the Gaussian RBF using -1/2 instead of -1 in the exponent */
                sim->data[i*n + j] = exp(-dist/2);
            }
            /* diagonal remains 0 as initialized by calloc */
        }
    }
    
    return sim;
}

matrix* calculate_diagonal_degree_matrix(matrix* sim_matrix) {
    int i, j;
    matrix* degree;
    
    degree = create_matrix(sim_matrix->rows, sim_matrix->rows);
    if (!degree) return NULL;
    
    for (i = 0; i < sim_matrix->rows; i++) {
        double sum = 0.0;
        for (j = 0; j < sim_matrix->rows; j++) {
            sum += sim_matrix->data[i * sim_matrix->rows + j];
        }
        degree->data[i * sim_matrix->rows + i] = sum;
    }
    
    return degree;
}

matrix* calculate_normalized_similarity(matrix* sim_matrix, matrix* degree_matrix) {
    int i, j;
    int n = sim_matrix->rows;
    matrix* norm;
    
    norm = create_matrix(n, n);
    if (!norm) return NULL;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double d_ii = sqrt(degree_matrix->data[i*n + i]);
            double d_jj = sqrt(degree_matrix->data[j*n + j]);
            if (d_ii > 0 && d_jj > 0) {
                norm->data[i*n + j] = sim_matrix->data[i*n + j] / (d_ii * d_jj);
            }
        }
    }
    
    return norm;
}

void matrix_multiply(matrix* result, matrix* mat1, matrix* mat2, int transpose2) {
    int i, j, k;
    int m = mat1->rows;
    int n = transpose2 ? mat2->rows : mat2->cols;
    int p = transpose2 ? mat2->cols : mat2->rows;
    double* temp;
    
    temp = (double*)calloc(m * n, sizeof(double));
    if (!temp) return;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < p; k++) {
                double val1 = mat1->data[i*p + k];
                double val2 = transpose2 ? mat2->data[j*p + k] : mat2->data[k*n + j];
                sum += val1 * val2;
            }
            temp[i*n + j] = sum;
        }
    }
    
    memcpy(result->data, temp, m * n * sizeof(double));
    free(temp);
}

double frobenius_norm_diff(matrix* mat1, matrix* mat2) {
    double sum = 0.0;
    int size = mat1->rows * mat1->cols;
    int i;
    
    for (i = 0; i < size; i++) {
        double diff = mat1->data[i] - mat2->data[i];
        sum += diff * diff;
    }
    
    return sqrt(sum);
}

matrix* optimize_h(matrix* w, matrix* h_init, int max_iter, double epsilon) {
    int n = w->rows;
    int k = h_init->cols;
    int iter, i, j;
    matrix* h_prev;
    matrix* h_curr;
    matrix* temp1;
    matrix* temp2;
    matrix* result;
    
    h_prev = create_matrix(n, k);
    h_curr = create_matrix(n, k);
    temp1 = create_matrix(n, k);
    temp2 = create_matrix(n, k);
    
    if (!h_prev || !h_curr || !temp1 || !temp2) {
        if (h_prev) free_matrix(h_prev);
        if (h_curr) free_matrix(h_curr);
        if (temp1) free_matrix(temp1);
        if (temp2) free_matrix(temp2);
        return NULL;
    }
    
    /* Copy initial H */
    memcpy(h_curr->data, h_init->data, n * k * sizeof(double));
    
    for (iter = 0; iter < max_iter; iter++) {
        /* Copy current H to previous H */
        memcpy(h_prev->data, h_curr->data, n * k * sizeof(double));
        
        /* Calculate WH */
        matrix_multiply(temp1, w, h_curr, 0);
        
        /* Calculate H(H^T H) */
        matrix_multiply(temp2, h_curr, h_curr, 1);
        matrix_multiply(temp2, temp2, h_curr, 0);
        
        /* Update H */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                if (temp2->data[i*k + j] > 0) {
                    h_curr->data[i*k + j] *= (1 - BETA + BETA * temp1->data[i*k + j] / temp2->data[i*k + j]);
                }
            }
        }
        
        /* Check convergence */
        if (frobenius_norm_diff(h_curr, h_prev) < epsilon) {
            break;
        }
    }
    
    result = create_matrix(n, k);
    if (result) {
        memcpy(result->data, h_curr->data, n * k * sizeof(double));
    }
    
    free_matrix(h_prev);
    free_matrix(h_curr);
    free_matrix(temp1);
    free_matrix(temp2);
    
    return result;
}

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
    
    /* Allocate memory and read points */
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

void print_matrix(matrix* mat) {
    int i, j;
    for (i = 0; i < mat->rows; i++) {
        for (j = 0; j < mat->cols; j++) {
            printf("%.4f", mat->data[i * mat->cols + j]);
            if (j < mat->cols - 1) printf(",");
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    char* goal;
    char* filename;
    int n, d;
    double* points;
    matrix *sim = NULL, *ddg = NULL, *norm = NULL;
    
    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    goal = argv[1];
    filename = argv[2];
    points = read_data(filename, &n, &d);
    if (!points) return 1;
    
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