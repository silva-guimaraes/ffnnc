
#include "../../ann.c"

// 784 step 64 step 10
// 784 step 64 soft 10
// 784 step 128 soft 10
// 784 relu 128 soft 10 (400)

int main(int argc, char** argv)
{
    printf("ffnnc\n");

    const layout nn_layout[] = {
        {784, STEP}, 
        {64, SOFTMAX}, 
        {10, NONE},
        {0, NONE}};

    struct network* nn = make_nn(nn_layout, 0.02);

    td_context tdc = {.ntrain = 2000, .ntest = 200, .test_freq = 10}; 
    parse_idx_files("./dataset//train-images.idx3-ubyte", "./dataset/train-labels.idx1-ubyte", 
            &tdc, training);
    parse_idx_files("./dataset/t10k-images.idx3-ubyte", "./dataset/t10k-labels.idx1-ubyte", 
            &tdc, testing);

    print_layout(nn);

    train_nn(nn, &tdc, 20); 
    export_model(nn, "example.ffnnc");

    free_nn(nn);

    return 0;
}

