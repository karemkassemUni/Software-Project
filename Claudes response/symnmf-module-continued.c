#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Helper function to convert Python list to Matrix */
static Matrix* PyList_ToMatrix(PyObject* list) {
    Py_ssize_t rows, cols;
    Py_ssize_t i, j;
    PyObject *row, *item;
    Matrix* matrix;
    
    if (!PyList_Check(list)) return NULL;
    
    rows = PyList_Size(list);
    if (rows == 0) return NULL;
    
    row = PyList_GetItem(list, 0);
    if (!PyList_Check(row)) return NULL;
    cols = PyList_Size(row);
    
    matrix = create_matrix(rows, cols);
    if (matrix == NULL) return NULL;
    
    for (i = 0; i < rows; i++) {
        row = PyList_GetItem(list, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols) {
            free_matrix(matrix);
            return NULL;
        }
        
        for (j = 0; j < cols; j++) {
            item = PyList_GetItem(row, j);
            matrix->data[i][j] = PyFloat_AsDouble(item);
        }
    }
    
    return matrix;
}

/* Helper function to convert Matrix to Python list */
static PyObject* Matrix_ToPyList(Matrix* matrix) {
    PyObject *list, *row;
    int i, j;
    
    list = PyList_New(matrix->rows);
    if (list == NULL) return NULL;
    
    for (i = 0; i < matrix->rows; i++) {
        row = PyList_New(matrix->cols);
        if (row == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        
        for (j = 0; j < matrix->cols; j++) {
            PyList_SET_ITEM(row, j, PyFloat_FromDouble(matrix->data[i][j]));
        }
        
        PyList_SET_ITEM(list, i, row);
    }
    
    return list;
}

/* Python callable functions */
static PyObject* symnmf_sym(PyObject* self, PyObject* args) {
    PyObject *points_list;
    Matrix *points, *similarity;
    PyObject *result;
    
    if (!PyArg_ParseTuple(args, "O", &points_list))
        return NULL;
    
    points = PyList_ToMatrix(points_list);
    if (points == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    similarity = calculate_similarity_matrix(points);
    free_matrix(points);
    
    if (similarity == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = Matrix_ToPyList(similarity);
    free_matrix(similarity);
    
    return result;
}

static PyObject* symnmf_ddg(PyObject* self, PyObject* args) {
    PyObject *points_list;
    Matrix *points, *similarity, *degree;
    PyObject *result;
    
    if (!PyArg_ParseTuple(args, "O", &points_list))
        return NULL;
    
    points = PyList_ToMatrix(points_list);
    if (points == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    similarity = calculate_similarity_matrix(points);
    free_matrix(points);
    
    if (similarity == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    degree = calculate_diagonal_degree_matrix(similarity);
    free_matrix(similarity);
    
    if (degree == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = Matrix_ToPyList(degree);
    free_matrix(degree);
    
    return result;
}

static PyObject* symnmf_norm(PyObject* self, PyObject* args) {
    PyObject *points_list;
    Matrix *points, *similarity, *degree, *normalized;
    PyObject *result;
    
    if (!PyArg_ParseTuple(args, "O", &points_list))
        return NULL;
    
    points = PyList_ToMatrix(points_list);
    if (points == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    similarity = calculate_similarity_matrix(points);
    free_matrix(points);
    
    if (similarity == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    degree = calculate_diagonal_degree_matrix(similarity);
    if (degree == NULL) {
        free_matrix(similarity);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    normalized = calculate_normalized_similarity(similarity, degree);
    free_matrix(similarity);
    free_matrix(degree);
    
    if (normalized == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = Matrix_ToPyList(normalized);
    free_matrix(normalized);
    
    return result;
}

static PyObject* symnmf_symnmf(PyObject* self, PyObject* args) {
    PyObject *h_init_list, *w_list;
    Matrix *h_init, *w, *result_matrix;
    PyObject *result;
    
    if (!PyArg_ParseTuple(args, "OO", &h_init_list, &w_list))
        return NULL;
    
    h_init = PyList_ToMatrix(h_init_list);
    w = PyList_ToMatrix(w_list);
    
    if (h_init == NULL || w == NULL) {
        if (h_init) free_matrix(h_init);
        if (w) free_matrix(w);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result_matrix = perform_symnmf(h_init, w, MAX_ITER, EPSILON);
    free_matrix(w);
    
    if (result_matrix == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    result = Matrix_ToPyList(result_matrix);
    free_matrix(result_matrix);
    
    return result;
}

/* Method definition table */
static PyMethodDef SysnmfMethods[] = {
    {"sym", symnmf_sym, METH_VARARGS, "Calculate similarity matrix"},
    {"ddg", symnmf_ddg, METH_VARARGS, "Calculate diagonal degree matrix"},
    {"norm", symnmf_norm, METH_VARARGS, "Calculate normalized similarity matrix"},
    {"symnmf", symnmf_symnmf, METH_VARARGS, "Perform symNMF clustering"},
    {NULL, NULL, 0, NULL}
};

/* Module definition structure */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    "Symmetric Non-negative Matrix Factorization implementation",
    -1,
    SysnmfMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
