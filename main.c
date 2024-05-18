#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int rows;
    int cols;
    double *values;
} array_t;

array_t *array_create(int rows, int cols) {
    array_t *array = calloc(1, sizeof(array_t));
    array->values = calloc(rows * cols, sizeof(double));
    array->rows = rows;
    array->cols = cols;
    return array;
}

array_t *read_parameters(char *filename, int num_rows, int num_cols) {
    FILE *fp;
    array_t *array = array_create(num_rows, num_cols);

    printf("Reading parameters from %s\n", filename);

    // Open file
    fp = fopen(filename, "r");
    // Read file and populate array
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            fscanf(fp, "%lf", &array->values[i * num_cols + j]);
            // printf("array->values[%d][%d] = %lf\n", i, j, array->values[i * num_cols + j]);
        }
    }
    // Close file
    fclose(fp);
    return array;
}

array_t *transpose(array_t *array) {
    array_t *transposed = array_create(array->cols, array->rows);

    for (int i = 0; i < array->rows; i++) {
        for (int j = 0; j < array->cols; j++) {
            transposed->values[j * array->rows + i] = array->values[i * array->cols + j];
        }
    }

    return transposed;
}

array_t *multiply(array_t *a, array_t *b) {
    array_t *c = array_create(a->rows, b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            for (int k = 0; k < a->cols; k++) {
                c->values[i * b->cols + j] += a->values[i * a->cols + k] * b->values[k * b->cols + j];
            }
        }
    }

    return c;
}

array_t *add(array_t *a, array_t *b) {
    array_t *c = array_create(a->rows, a->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->values[i * a->cols + j] = a->values[i * a->cols + j] + b->values[i * a->cols + j];
        }
    }

    return c;
}

int argmax(array_t *array) {
    // Expect array to be 1D with 1 row and multiple columns
    int max_index = 0;
    double max_value = array->values[0];

    for (int i = 0; i < array->cols; i++) {
        if (array->values[i] > max_value) {
            max_index = i;
            max_value = array->values[i];
        }
    }

    return max_index;
}

void print_array_shape(array_t *array) {
    printf("rows: %d, cols: %d\n", array->rows, array->cols);
}

void array_destroy(array_t *array) {
    free(array->values);
    free(array);
    array->values = NULL;
    array = NULL;
    return;
}

int main(void) {
    array_t *dense_1_weight = read_parameters("data/export_dense_1.weight", 784, 64);
    array_t *dense_2_weight = read_parameters("data/export_dense_2.weight", 64, 64);
    array_t *dense_3_weight = read_parameters("data/export_dense_3.weight", 64, 10);
    array_t *dense_1_bias = read_parameters("data/export_dense_1.bias", 64, 1);
    array_t *dense_2_bias = read_parameters("data/export_dense_2.bias", 64, 1);
    array_t *dense_3_bias = read_parameters("data/export_dense_3.bias", 10, 1);

    array_t *inputs = array_create(1, 784);

    array_t *a = multiply(inputs, dense_1_weight);
    array_t *t_dense_1_bias = transpose(dense_1_bias);
    array_t *dense_1_output = add(a, t_dense_1_bias);
    array_t *b = multiply(dense_1_output, dense_2_weight);
    array_t *t_dense_2_bias = transpose(dense_2_bias);
    array_t *dense_2_output = add(b, t_dense_2_bias);
    array_t *c = multiply(dense_2_output, dense_3_weight);
    array_t *t_dense_3_bias = transpose(dense_3_bias);
    array_t *dense_3_output = add(c, t_dense_3_bias);

    int max_index = argmax(dense_3_output);

    printf("Predicted digit: %d\n", max_index);

    array_destroy(inputs);

    array_destroy(dense_3_output);
    array_destroy(dense_2_output);
    array_destroy(dense_1_output);
    array_destroy(t_dense_3_bias);
    array_destroy(t_dense_2_bias);
    array_destroy(t_dense_1_bias);
    array_destroy(c);
    array_destroy(b);
    array_destroy(a);

    array_destroy(dense_1_bias);
    array_destroy(dense_2_bias);
    array_destroy(dense_3_bias);
    array_destroy(dense_1_weight);
    array_destroy(dense_2_weight);
    array_destroy(dense_3_weight);

    return 0;
}
