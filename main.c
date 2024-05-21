#include "ctensor_arena.h"
#include "ctensor_tensor.h"
#include "ctensor_ops.h"

#include <stdio.h>
#include <stdio.h>

ctensor_tensor_t *__create_tensor_from_file(ctensor_arena_t *arena, const char *filename, uint64_t num_rows, uint64_t num_cols) {
    FILE *fp;
    ctensor_tensor_t *array = ctensor_create_tensor(arena, (uint64_t[]) { num_rows, num_cols }, 2);

    printf("Reading parameters from %s\n", filename);

    // Open file
    fp = fopen(filename, "r");
    // Read file and populate array
    for (uint64_t i = 0; i < num_rows; ++i) {
        for (uint64_t j = 0; j < num_cols; ++j) {
            fscanf(fp, "%f", &array->data[i * num_cols + j]);
        }
    }
    // Close file
    fclose(fp);

    return array;
}

int main(void) {
    ctensor_arena_t *arena = ctensor_arena_create(400000);

    ctensor_tensor_t *w1 = __create_tensor_from_file(arena, "./data/export_dense_1.weight", 784, 64);
    ctensor_tensor_t *b1 = __create_tensor_from_file(arena, "./data/export_dense_1.bias", 64, 1);
    ctensor_tensor_t *w2 = __create_tensor_from_file(arena, "./data/export_dense_2.weight", 64, 10);
    ctensor_tensor_t *b2 = __create_tensor_from_file(arena, "./data/export_dense_2.bias", 10, 1);

    ctensor_tensor_t *input = ctensor_create_tensor(arena, (uint64_t[]) { 28, 28 }, 2);
    ctensor_tensor_t *flattened = ctensor_flatten(arena, input);
    ctensor_tensor_t *net_input = ctensor_permute(arena, flattened, (uint64_t[]) { 1, 0 }); // [1, 784]

    ctensor_tensor_t *a1 = ctensor_multiply(arena, net_input, w1, 1, 0); // [1, 64]
    ctensor_tensor_t *t_b1 = ctensor_permute(arena, b1, (uint64_t[]) { 1, 0 }); // [1, 64]
    ctensor_tensor_t *z1 = ctensor_add(arena, a1, t_b1); // [1, 64]
    ctensor_tensor_t *o1 = ctensor_relu(arena, z1); // [1, 64]
    ctensor_tensor_t *a2 = ctensor_multiply(arena, o1, w2, 1, 0); // [1, 10]
    ctensor_tensor_t *t_b2 = ctensor_permute(arena, b2, (uint64_t[]) { 1, 0 }); // [1, 10]
    ctensor_tensor_t *z2 = ctensor_add(arena, a2, t_b2); // [1, 10]

    ctensor_print_tensor(z2);

    uint64_t prediction = 0;

    for (uint64_t i = 0; i < 10; ++i) {
        if (z2->data[i] > z2->data[prediction]) {
            prediction = i;
        }
    }

    printf("Prediction: %llu\n", prediction);

    ctensor_arena_destroy(arena);
    return 0;
}
