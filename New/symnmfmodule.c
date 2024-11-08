#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

static PyObject* create_py_matrix_from_c(matrix* c_matrix) {
    PyObject* py_matrix;
    int i, j;
    
    if (!c_matrix) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    py_matrix = PyList_New(c_matrix->rows);
    if (!py_matrix) {
        return NULL;
    }
    
    for (i = 0; i < c_matrix->rows; i++) {
        PyObject* row = PyList_New(c_matrix->cols);
        if (!row) {
            Py_DECREF(py_matrix);
            return NULL;
        }
        
        for (j = 0; j < c_matrix->cols; j++) {
            PyObject* val = PyFloat_FromDouble(c_matrix->data[i * c_matrix->cols + j]);
            if (!val) {
                Py_DECREF(row);
                Py_DECREF(py_matrix);
                return NULL;
            }
            PyList_SET_ITEM(row, j, val);
        }
        PyList_SET_ITEM(py_matrix, i, row);
    }
    
    return py_matrix;
}

static double* convert_py_points_to_c(PyObject* points_list, int n, int d) {
    double* c_points;
    int i, j;
    
    c_points = (double*)calloc(n * d, sizeof(double));
    if (!c_points) {
        return NULL;
    }
    
    for (i = 0; i < n; i++) {
        PyObject* row = PyList_GetItem(points_list, i);
        for (j = 0; j < d; j++) {
            PyObject* item = PyList_GetItem(row, j);
            c_points[i * d + j] = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                free(c_points);
                return NULL;
            }
        }
    }
    
    return c_points;
}

static matrix* convert_py_matrix_to_c(PyObject* py_matrix, int rows, int cols) {
    matrix* c_matrix;
    int i, j;
    
    c_matrix = create_matrix(rows, cols);
    if (!c_matrix) {
        return NULL;
    }
    
    for (i = 0; i < rows; i++) {
        PyObject* row = PyList_GetItem(py_matrix, i);
        for (j = 0; j < cols; j++) {
            PyObject* item = PyList_GetItem(row, j);
            c_matrix->data[i * cols + j] = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                free_matrix(c_matrix);
                return NULL;
            }
        }
    }
    
    return c_matrix;
}

static PyObject* symnmf_sym(PyObject* self, PyObject* args) {
    PyObject* points_list;
    int n, d;
    double* points;
    matrix* sim;
    PyObject* result;
    
if (!PyArg_ParseTuple(args, "Oii", &points_list, &n, &d)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    points = convert_py_points_to_c(points_list, n, d);
    if (!points) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    sim = calculate_similarity_matrix(points, n, d);
    free(points);
    if (!sim) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = create_py_matrix_from_c(sim);
    free_matrix(sim);
    
    return result;
}

static PyObject* symnmf_ddg(PyObject* self, PyObject* args) {
    PyObject* sim_list;
    int n;
    matrix* sim;
    matrix* ddg;
    PyObject* result;
    
    if (!PyArg_ParseTuple(args, "Oi", &sim_list, &n)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    sim = convert_py_matrix_to_c(sim_list, n, n);
    if (!sim) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    ddg = calculate_diagonal_degree_matrix(sim);
    free_matrix(sim);
    if (!ddg) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = create_py_matrix_from_c(ddg);
    free_matrix(ddg);
    
    return result;
}

static PyObject* symnmf_norm(PyObject* self, PyObject* args) {
    PyObject* sim_list;
    PyObject* ddg_list;
    int n;
    matrix* sim;
    matrix* ddg;
    matrix* norm;
    PyObject* result;
    
    if (!PyArg_ParseTuple(args, "OOi", &sim_list, &ddg_list, &n)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    sim = convert_py_matrix_to_c(sim_list, n, n);
    if (!sim) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    ddg = convert_py_matrix_to_c(ddg_list, n, n);
    if (!ddg) {
        free_matrix(sim);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    norm = calculate_normalized_similarity(sim, ddg);
    free_matrix(sim);
    free_matrix(ddg);
    if (!norm) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = create_py_matrix_from_c(norm);
    free_matrix(norm);
    
    return result;
}

static PyObject* symnmf_factorize(PyObject* self, PyObject* args) {
    PyObject* w_list;
    PyObject* h_init_list;
    int n, k;
    matrix* w;
    matrix* h_init;
    matrix* h;
    PyObject* result;
    
    if (!PyArg_ParseTuple(args, "OOii", &w_list, &h_init_list, &n, &k)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    w = convert_py_matrix_to_c(w_list, n, n);
    if (!w) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    h_init = convert_py_matrix_to_c(h_init_list, n, k);
    if (!h_init) {
        free_matrix(w);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    h = optimize_h(w, h_init, MAX_ITER, EPSILON);
    free_matrix(w);
    free_matrix(h_init);
    if (!h) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = create_py_matrix_from_c(h);
    free_matrix(h);
    
    return result;
}

static PyMethodDef SymnmfMethods[] = {
    {"sym", symnmf_sym, METH_VARARGS, "Calculate similarity matrix"},
    {"ddg", symnmf_ddg, METH_VARARGS, "Calculate diagonal degree matrix"},
    {"norm", symnmf_norm, METH_VARARGS, "Calculate normalized similarity matrix"},
    {"factorize", symnmf_factorize, METH_VARARGS, "Perform matrix factorization"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    SymnmfMethods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}