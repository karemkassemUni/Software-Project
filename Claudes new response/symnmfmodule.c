#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Helper function to convert C matrix to Python list */
static PyObject* matrix_to_python(matrix* c_matrix) {
    PyObject *python_matrix, *row, *value;
    int i, j;
    
    if (!c_matrix) {
        return NULL;
    }
    
    python_matrix = PyList_New(c_matrix->rows);
    if (!python_matrix) {
        return NULL;
    }
    
    for (i = 0; i < c_matrix->rows; i++) {
        row = PyList_New(c_matrix->cols);
        if (!row) {
            Py_DECREF(python_matrix);
            return NULL;
        }
        
        for (j = 0; j < c_matrix->cols; j++) {
            value = PyFloat_FromDouble(c_matrix->data[i * c_matrix->cols + j]);
            if (!value) {
                Py_DECREF(row);
                Py_DECREF(python_matrix);
                return NULL;
            }
            PyList_SET_ITEM(row, j, value);
        }
        PyList_SET_ITEM(python_matrix, i, row);
    }
    
    return python_matrix;
}

/* Helper function to convert Python points to C array */
static double* points_to_c(PyObject* points_list, int n, int d) {
    double* c_points;
    PyObject *row, *item;
    int i, j;
    
    c_points = (double*)calloc(n * d, sizeof(double));
    if (!c_points) {
        return NULL;
    }
    
    for (i = 0; i < n; i++) {
        row = PyList_GetItem(points_list, i);
        if (!row) {
            free(c_points);
            return NULL;
        }
        
        for (j = 0; j < d; j++) {
            item = PyList_GetItem(row, j);
            if (!item) {
                free(c_points);
                return NULL;
            }
            
            c_points[i * d + j] = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                free(c_points);
                return NULL;
            }
        }
    }
    
    return c_points;
}

/* Helper function to convert Python matrix to C matrix */
static matrix* matrix_to_c(PyObject* python_matrix, int rows, int cols) {
    matrix* c_matrix;
    PyObject *row, *item;
    int i, j;
    
    c_matrix = create_matrix(rows, cols);
    if (!c_matrix) {
        return NULL;
    }
    
    for (i = 0; i < rows; i++) {
        row = PyList_GetItem(python_matrix, i);
        if (!row) {
            free_matrix(c_matrix);
            return NULL;
        }
        
        for (j = 0; j < cols; j++) {
            item = PyList_GetItem(row, j);
            if (!item) {
                free_matrix(c_matrix);
                return NULL;
            }
            
            c_matrix->data[i * cols + j] = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                free_matrix(c_matrix);
                return NULL;
            }
        }
    }
    
    return c_matrix;
}

/* Python interface for similarity matrix calculation */
static PyObject* sym_calc(PyObject* self, PyObject* args) {
    PyObject *points_list, *result;
    int n, d;
    double* points;
    matrix* sim_matrix;
    
    if (!PyArg_ParseTuple(args, "Oii", &points_list, &n, &d)) {
        return NULL;
    }
    
    points = points_to_c(points_list, n, d);
    if (!points) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    sim_matrix = calculate_similarity_matrix(points, n, d);
    free(points);
    
    if (!sim_matrix) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = matrix_to_python(sim_matrix);
    free_matrix(sim_matrix);
    
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    return result;
}

/* Python interface for diagonal degree matrix calculation */
static PyObject* ddg_calc(PyObject* self, PyObject* args) {
    PyObject *sim_list, *result;
    int n;
    matrix *sim_matrix, *ddg_matrix;
    
    if (!PyArg_ParseTuple(args, "Oi", &sim_list, &n)) {
        return NULL;
    }
    
    sim_matrix = matrix_to_c(sim_list, n, n);
    if (!sim_matrix) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    ddg_matrix = calculate_diagonal_degree_matrix(sim_matrix);
    free_matrix(sim_matrix);
    
    if (!ddg_matrix) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = matrix_to_python(ddg_matrix);
    free_matrix(ddg_matrix);
    
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    return result;
}

/* Python interface for normalized similarity matrix calculation */
static PyObject* norm_calc(PyObject* self, PyObject* args) {
    PyObject *sim_list, *ddg_list, *result;
    int n;
    matrix *sim_matrix, *ddg_matrix, *norm_matrix;
    
    if (!PyArg_ParseTuple(args, "OOi", &sim_list, &ddg_list, &n)) {
        return NULL;
    }
    
    sim_matrix = matrix_to_c(sim_list, n, n);
    if (!sim_matrix) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    ddg_matrix = matrix_to_c(ddg_list, n, n);
    if (!ddg_matrix) {
        free_matrix(sim_matrix);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    norm_matrix = calculate_normalized_similarity(sim_matrix, ddg_matrix);
    free_matrix(sim_matrix);
    free_matrix(ddg_matrix);
    
    if (!norm_matrix) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = matrix_to_python(norm_matrix);
    free_matrix(norm_matrix);
    
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    return result;
}

/* Python interface for matrix factorization */
static PyObject* factorize_calc(PyObject* self, PyObject* args) {
    PyObject *w_list, *h_init_list, *result;
    int n, k;
    matrix *w_matrix, *h_init_matrix, *h_final_matrix;
    
    if (!PyArg_ParseTuple(args, "OOii", &w_list, &h_init_list, &n, &k)) {
        return NULL;
    }
    
    w_matrix = matrix_to_c(w_list, n, n);
    if (!w_matrix) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    h_init_matrix = matrix_to_c(h_init_list, n, k);
    if (!h_init_matrix) {
        free_matrix(w_matrix);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    h_final_matrix = optimize_h(w_matrix, h_init_matrix, MAX_ITER, EPSILON);
    free_matrix(w_matrix);
    free_matrix(h_init_matrix);
    
    if (!h_final_matrix) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = matrix_to_python(h_final_matrix);
    free_matrix(h_final_matrix);
    
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    return result;
}

/* Method definitions */
static PyMethodDef symnmf_methods[] = {
    {"sym", sym_calc, METH_VARARGS, "Calculate similarity matrix"},
    {"ddg", ddg_calc, METH_VARARGS, "Calculate diagonal degree matrix"},
    {"norm", norm_calc, METH_VARARGS, "Calculate normalized similarity matrix"},
    {"factorize", factorize_calc, METH_VARARGS, "Perform matrix factorization"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    symnmf_methods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}