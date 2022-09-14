#include <stdio.h>
#include <string.h>
#include <math.h>
#include <endian.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <ncurses.h>

//https://github.com/casperbh96/Neural-Network-From-Scratch/blob/master/NN_From_Scratch.ipynb

#define MAX_MNIST 100
#define TRAIN_IMAGES_PATH "/home/xi/Desktop/prog/lisp/common lisp/ann/data/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "/home/xi/Desktop/prog/lisp/common lisp/ann/data/train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH "/home/xi/Desktop/prog/lisp/common lisp/ann/data/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "/home/xi/Desktop/prog/lisp/common lisp/ann/data/t10k-labels.idx1-ubyte"

#define W1 weights[0]
#define W2 weights[1]
#define W3 weights[2]

struct matrix {
    long double** mat; // array de ponteiros. cada ponteiro com uma array (de long double) diferente.
    size_t m;
    size_t n;
};

typedef struct matrix matrix;
const size_t sdo = sizeof(long double); 
const long double L_RATE = 0.001; //não abusar dessa variavel.

struct mnist {
    uint8_t* image;
    uint8_t label;
    size_t rows, columns;
    matrix* act0, *act1, *act2, *act3;
    matrix* z1, *z2, *z3;
    struct mnist* next;
};


#define mat_iterator(mat, y) \
    for (size_t i = 0; i < mat->m; i++){ for (size_t j = 0; j < mat->n; j++){ y } }

#define PRINT_MAT_LIMIT

void print_mat(matrix* a, bool limit) //debug
{

    printf("\n%ld x %ld\n", a->m, a->n); 
    for (size_t i = 0; i < a->m; i++){
	if (limit)
	    printf("| "); 
	for (size_t j = 0; j < a->n; j++)
	{
	    if (!limit && j > 10){ 
		printf("..%ld..\t%.10LF\t", a->n - j, a->mat[i][a->n - 1]);
		break;
	    } 
            printf("%.10LF\t", a->mat[i][j]); 
	}
	if (limit)
	    printf("|\n"); 
    } 
    if (!limit)
	printf("\n");
} 
long double** alloc_mat(size_t m, size_t n, long double init)
{ 
    long double** ret = malloc(m * sizeof(long double*));

    for (size_t i = 0; i < m ; i++) { 
	ret[i] = malloc(n * sdo);
	for (size_t j = 0; j < n; j++) ret[i][j] = init;
    } 
    return ret;
} 
matrix* new_mat_struct(size_t m, size_t n, long double** mat)
{
    matrix* ret = malloc(sizeof(struct matrix));

    ret->mat = mat;
    ret->m = m;
    ret->n = n;

    return ret; 
} 
matrix* new_mat(size_t m, size_t n, long double init)
{
    return new_mat_struct(m, n, (alloc_mat(m, n, init))); 
} 
matrix* copy_mat(matrix* x)
{
    long double** retmat = malloc(x->m * sdo);

    for (size_t i = 0; i < x->m ; i++) { 
	retmat[i] = malloc(x->n * sdo);
	for (size_t j = 0; j < x->n; j++) retmat[i][j] = x->mat[i][j];
    } 

    return new_mat_struct(x->m, x->n, retmat); 
}
void free_mat_arrays(long double** x, size_t m)
{
    for (size_t i = 0; i < m; i++){
	free(x[i]); 
    } 
    free(x);
}
void free_mat_struct(matrix* m)
{
    free_mat_arrays(m->mat, m->m);
    //free(m->mat);
    free(m); 
}
void transpose(matrix* a)
{ 
    long double** retmat = alloc_mat(a->n, a->m, 0);

    mat_iterator(a, retmat[j][i] = a->mat[i][j];);

    // trocar matrizes
    free_mat_arrays(a->mat, a->m); 
    size_t tmp = a->m;
    a->m = a->n;
    a->n = tmp;
    a->mat = retmat;
}
matrix* p_transpose(matrix* a)
{ 
    long double** retmat = alloc_mat(a->n, a->m, 0);

    mat_iterator(a, retmat[j][i] = a->mat[i][j];);

    return new_mat_struct(a->n, a->m, retmat);
}
matrix* dot_product(matrix* a, matrix* b)
{
    // tratar vetor como matriz coluna em casos de multiplição entre uma matriz em um vetor  
    if (b->m == 1 && a->n == b->n){ 
	transpose(b);
    }
    assert(a->n == b->m);
    matrix* ret = new_mat(a->m, b->n, 0); 

    for (size_t i = 0; i < a->m; i++)
    { 
	for (size_t j = 0; j < b->n; j++){
	    long double sum = 0;
	    for (size_t k = 0; k < b->m; k++)
		sum += (a->mat[i][k] * b->mat[k][j]); 
	    ret->mat[i][j] = sum;
	}
    }
    return ret; 
} 
//matrix* identity(size_t s)
//{
//    long double** mat = alloc_mat(s, s, 0);
//
//    for (size_t i = 0; i < s; i++) 
//	mat[i][i] = 1;
//
//    return new_mat_struct(s, s, mat); 
//} 
void flatten(matrix* a)
{
    size_t snew = a->m * a->n;
    long double** ret = alloc_mat(1, snew, 0);

    mat_iterator(a, ret[0][j + (i * a->n)] = a->mat[i][j];);
    
    free_mat_arrays(a->mat, a->m);
    a->mat = ret;
    a->m = 1;
    a->n = snew;
}
matrix* outer(matrix* a, matrix* b)
{
    if (a->m > 1) flatten(a);
    if (b->m > 1) flatten(b);

    matrix* ret = new_mat(a->n, b->n, 0);
    for (size_t i = 0; i < a->n; i++){
	for (size_t j = 0; j < b->n; j++) 
	    ret->mat[i][j] = a->mat[0][i] * b->mat[0][j]; 
    } 
    return ret; 
}
void normalize(struct mnist* x)
{
    for (size_t i = 0; i < x->rows * x->columns; i++)
	x->act0->mat[0][i] = (long double) x->image[i] / 255; 
} 
//http://yann.lecun.com/exdb/mnist/
struct mnist* parse_idx_files(char* x, char* y)
{
    FILE* images = fopen(x, "r"); assert(images != NULL);
    FILE* labels = fopen(y, "r"); assert(labels != NULL);
    uint32_t magic, img_count, rows, columns,
	y_magic, label_count; 

#define read_u32be(x, y)\
    assert(fread(&x, 4, 1, y) != 0); x = htobe32(x); 

    read_u32be(magic, images); assert(magic == 2051);
    read_u32be(img_count, images);
    read_u32be(rows, images);
    read_u32be(columns, images);

    read_u32be(y_magic, labels); assert(y_magic == 2049);
    read_u32be(label_count, labels);

    assert(img_count == label_count);

    struct mnist* first, *p = &(struct mnist) { 0 };
    first = p;

    for (uint32_t i = 0; i < img_count && i < MAX_MNIST; i++)
    {
	struct mnist* tmp = malloc(sizeof(struct mnist));
	tmp->image = malloc(rows * columns);
	assert(fread(tmp->image, 1, rows * columns, images) != 0); 
	assert(fread(&tmp->label, 1, 1, labels) != 0);
	tmp->next = NULL; 
	tmp->act0 = NULL;
	tmp->rows = rows;
	tmp->columns = columns;
	tmp->act0 = new_mat(1, rows * columns, 0);

	p->next = tmp;
	p = p->next; 
    }

    fclose(images); fclose(labels);

    return first->next; 
} 
void free_activations(struct mnist* x) 
{
    free_mat_struct(x->act1);
    free_mat_struct(x->act2);
    free_mat_struct(x->act3);
    free_mat_struct(x->z1);
    free_mat_struct(x->z2);
    free_mat_struct(x->z3);
}
void free_mnist(struct mnist* x)
{
    free(x->image);
    free_activations(x);
    free_mat_struct(x->act0);
    free(x);

}
matrix* one_hot(uint8_t x)
{
    assert(x < 10);

    matrix* ret = new_mat(10, 1, 0);

    ret->mat[x][0] = 1;

    return ret;
} 
long double sum_mat(matrix* x)
{
    long double sum = 0;
    mat_iterator(x, sum += x->mat[i][j];);
    return sum; 
}
matrix* softmax(matrix* x, bool derivative)
{ 
    matrix* exps = copy_mat(x);

    { //argmax 
	long double max = x->mat[0][0]; 
	mat_iterator(x, if (x->mat[i][j] > max) max = x->mat[i][j];); 

	mat_iterator(exps, exps->mat[i][j] = expl(exps->mat[i][j] - max););
    }

    long double sum = sum_mat(exps); 

    if (derivative) { 
	mat_iterator(exps, exps->mat[i][j] = (exps->mat[i][j] / sum) * (1 - exps->mat[i][j] / sum););
    }
    else { 
	mat_iterator(exps, exps->mat[i][j] /= sum;); 
    } 

    return exps; 
} 
long double sigmoid(long double z, bool derivative)
{
    if (derivative)
	return (expl(-z)) / (powl((expl(-z) + 1), 2.0));
    else 
	return 1 / (1 + expl(-z));
} 
matrix* sigmoid_activation(matrix* z, bool derivative)
{
    matrix* ret = new_mat(z->m, z->n, 0); 

    mat_iterator(z, ret->mat[i][j] = sigmoid(z->mat[i][j], derivative););

    return ret;
}
void forward_pass(struct mnist* x, matrix* weights[3])
{
    if (x->act0 == NULL) // normalizar todos os valores da input layer pra  0, 1, ou entre 0 e 1 antes da primeira ativação
	normalize(x); 

    x->z1 = dot_product(W1, x->act0);
    x->act1 = sigmoid_activation(x->z1, false);

    x->z2 = dot_product(W2, x->act1);
    x->act2 = sigmoid_activation(x->z2, false);

    x->z3 = dot_product(W3, x->act2);
    x->act3 = softmax(x->z3, false);
} 
matrix** backward_pass(struct mnist* x, matrix* weights[3])
{
    matrix** gradients = malloc(sizeof(struct matrix*) * 3), 
	*error3, *error2, *error1; 
	

    {//computar atualização do peso 3 	
	error3 = copy_mat(x->act3);
	matrix* z3s = softmax(x->z3, true), *ohl = one_hot(x->label);
	mat_iterator(error3, 
		error3->mat[i][j] -= ohl->mat[i][j];
		error3->mat[i][j] *= 2;
		error3->mat[i][j] /= x->act3->m;
		error3->mat[i][j] *= z3s->mat[i][j]; 
		); 
	free_mat_struct(z3s); free_mat_struct(ohl); 

	gradients[2] = outer(error3, x->act2);
    } 

    { //computar atualização do peso 2 	
        matrix* w3t = copy_mat(W3); transpose(w3t);
        error2 = dot_product(w3t, error3);
        matrix* sigtmp = sigmoid_activation(x->z2, true);
        mat_iterator(error2, error2->mat[i][j] *= sigtmp->mat[i][j];);
        free_mat_struct(w3t); free_mat_struct(sigtmp);

        gradients[1] = outer(error2, x->act1);
    }

    { //computar atualização do peso 1 	
        matrix* w2t = copy_mat(W2); transpose(w2t);
        error1 = dot_product(w2t, error2);
        matrix* sigtmp = sigmoid_activation(x->z1, true);
        mat_iterator(error1, error1->mat[i][j] *= sigtmp->mat[i][j];);
        free_mat_struct(w2t); free_mat_struct(sigtmp);

        gradients[0] = outer(error1, x->act0);
    }
    free_mat_struct(error3); free_mat_struct(error2); free_mat_struct(error1);

    return gradients;
}
void update_parameters(matrix* gradients[3], matrix* weights[3])
{ 
    for (int k = 0; k < 3; k++){
	mat_iterator(weights[k], 
		gradients[k]->mat[i][j] *= L_RATE;
		weights[k]->mat[i][j] -= gradients[k]->mat[i][j];); 
	free_mat_struct(gradients[k]);
    }
    free(gradients); 
}
long double rand_ld(long double max)
{
    return ((long double) rand() / (long double) max) * 0.0245e-8;
}
int compute_acc(struct mnist* x)
{
    size_t argmax = 0; 
    mat_iterator(x->act3, if (x->act3->mat[i][j] >= x->act3->mat[argmax][0]) argmax = i;);

    if (argmax == x->label)
	return 1;
    else
	return 0;
}
void test_acc(struct mnist* x, matrix** weights)
{
    unsigned int i = 0, j = 0;
    for (; x != NULL; x = x->next, j++)
    { 
	forward_pass(x, weights);
	i += compute_acc(x);
	free_activations(x);
	//printf("teste: %d\n", j + 1);
    }
    printf("forward-pass teste: %.2f%%\n", ((float) i / j) * 100);
}
matrix* setup_weight(size_t m, size_t n, long double k)
{
    matrix* x = new_mat(m, n, 0);

    mat_iterator(x, x->mat[i][j] = rand_ld(k) - rand_ld(k);); 
    
    return x;
}
void dump_matrix(FILE* path, matrix* x)
{
    /* 4 bytes (big-endian) representando o numero de fileiras da matriz (m)
     * 4 bytes (big-endian) representando o numero de colunas da matriz (n)
     * m x n long doubles em ordem representando a matriz
     * não existe nenhuma separação entre matrizes em um arquivo com mais de uma matriz
     */
    fwrite(&x->m, sizeof(size_t), 1, path);
    fwrite(&x->n, sizeof(size_t), 1, path);

    for (size_t i = 0; i < x->m; i++){
	fwrite(x->mat[i], sizeof(long double), x->n, path);
    } 
}
void dump_params(matrix* weights[3])
{
    FILE* dump = fopen("c_ann.weights", "w"); assert(dump != NULL);

    for (int i = 0 ; i < 3; i++){
	dump_matrix(dump, weights[i]);
    }

    fclose(dump); 
}
matrix* load_matrix(FILE* load)
{
    size_t m, n; 
    fread(&m, sizeof(size_t), 1, load);
    fread(&n, sizeof(size_t), 1, load);

    matrix* ret = new_mat(m, n, 0); 
    mat_iterator(ret, 
	    fread(&ret->mat[i][j], sizeof(long double), 1,  load);); 

    return ret;

}
matrix** load_params(FILE* load)
{ 
    printf("load!\n");
    matrix** weights = malloc(sizeof(struct matrix*) * 3);

    for (int i = 0 ; i < 3; i++){
	/* se as matrizes tiverem sido salvas corretamente, o ponteiro da stream vai estar posicionado 
	 * no inicio de uma nova matriz toda vez que load_matrix retornar */ 
	weights[i] = load_matrix(load);
    } 
    return weights;
}

/* imagem (28x28)
 * 	v
 * input layer (784)
 * 	v
 * ativação sigmoid
 * 	v
 * primeira hidden layer (128) 
 * 	v
 * ativação sigmoid
 * 	v
 * segunda hidden layer (64) 
 * 	v
 * ativação softmax
 * 	v
 * output (10)
 */

#define INPUT_N 784
#define HIDDEN_1 128
#define HIDDEN_2 64
#define OUTPUT_N 10




int main(void)
{
    srand(time(NULL)); 
    size_t epochs = 10; 

    matrix** gradients;
    FILE* load = NULL; 
    //FILE* load = fopen("c_ann.weights", "r"); 
    /* 200 iterações:
     * LEAK SUMMARY:
     * ==15619==    definitely lost: 72 bytes in 3 blocks 
     * ==15619==    indirectly lost: 1,748,560 bytes in 205 blocks */
    matrix** weights = (load != NULL) ? load_params(load) : malloc(sizeof(struct matrix*) * 3); 
    //fclose(load); 

    W1 = setup_weight(HIDDEN_1, INPUT_N, 6);
    W2 = setup_weight(HIDDEN_2, HIDDEN_1, 6);
    W3 = setup_weight(OUTPUT_N, HIDDEN_2, 6); 

    struct mnist* train_set = parse_idx_files(TRAIN_IMAGES_PATH,TRAIN_LABELS_PATH); 
    printf("set de treinamento carregado com sucesso\n");
    struct mnist* test_set = parse_idx_files(TEST_IMAGES_PATH,TEST_LABELS_PATH); 
    printf("set de testes carregado com sucesso\n");

    printf("iniciando main loop...\n");
    print_mat(W3, false);
    for (unsigned int i = 0; i < epochs; i++) // epochs
    {  
	struct mnist* tmp = train_set; 
	printf("epoch %d...\n", i + 1);

	for (int j = 0; tmp != NULL; tmp = tmp->next, j++)
	{ 
	    forward_pass(tmp, weights);
	    gradients = backward_pass(tmp, weights);
	    if (tmp->next == NULL && i == epochs - 1)
		print_mat(gradients[2], false);
	    update_parameters(gradients, weights); 

	    free_activations(tmp); 
	} 
	printf("finalizado. computando accuracy..\n");
	test_acc(test_set, weights);
	printf("-------\n");
    }

#define free_dataset(x)\
    while (x != NULL) {\
	struct mnist* tmp = x;\
	x = x->next;\
	free_mnist(tmp);\
    }

    //free_dataset(train_set);
    //free_dataset(test_set);
    dump_params(weights);
    free_mat_struct(W1);
    free_mat_struct(W2);
    free_mat_struct(W3);
    free(weights);


}
