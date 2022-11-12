
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <endian.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>


struct matrix {
  long double** mat; // array de ponteiros
  size_t m;
  size_t n;
};
typedef struct matrix matrix;

struct layer {
  // ativação
  struct matrix* a;
  struct matrix* wgt; // pesos. a magia acontece aqui
  struct matrix* z; // pesos aplicados
  struct matrix* b;
  size_t size;
  // função de ativação para a proxuma camada
  matrix* (*act)(matrix*);
};
typedef struct layer layer;

struct network {
  struct layer* l;
  size_t amount;
};

struct layout {
  int size;
  matrix* (*act)(matrix*);
};
typedef struct layout layout;


const int end_nn = -1;
const size_t sdo = sizeof(long double); 
const long double L_RATE = 0.0005; //não abusar dessa variavel.


#define mat_iterator(mat, body)                 \
  for (size_t i = 0; i < mat->m; i++){          \
    for (size_t j = 0; j < mat->n; j++){        \
      body;                                     \
        }                                       \
  }

#define layer_iterator(nn, body)                \
  for (size_t i = 0; i < nn->amount; i++){      \
    body;                                       \
  }

#define PRINT_MAT_LIMIT


//printf("\n%ld x %ld\n", a->m, a->n);
#define print_mat(a, limit, limit_num, x, y){               \
    for (size_t ip = 0; ip < a->m; ip++){                   \
      printf("| ");                                         \
      for (size_t jp = 0; jp < a->n; jp++) {                \
        if (!limit && jp > limit_num){                      \
          printf(x, a->n - jp - 1, a->mat[ip][a->n - 1]);   \
          break;                                            \
        }                                                   \
        printf(y, a->mat[ip][jp]);                          \
      }                                                     \
      printf("|\n");                                        \
    }                                                       \
  } 

// matrix* a, bool limit
#define print_simple_mat(a, limit)\
  print_mat(a, limit, 8, "..%ld..\t%.2LF " , "%.2LF  ")

#define print_long_mat(a, limit)\
  print_mat(a, limit, 3, "..%ld..\t%.10LF\t" , "%.10LF\t")

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
    for (size_t j = 0; j < x->n; j++)
      retmat[i][j] = x->mat[i][j];
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

// void normalize(struct mnist* x)
// {
//   for (size_t i = 0; i < x->rows * x->columns; i++)
//     x->act0->mat[0][i] = (long double) x->image[i] / 255; 
// } 

//http://yann.lecun.com/exdb/mnist/
// struct mnist* parse_idx_files(char* x, char* y)
// {
//   FILE* images = fopen(x, "r"); assert(images != NULL);
//   FILE* labels = fopen(y, "r"); assert(labels != NULL);
//   uint32_t magic, img_count, rows, columns,
//     y_magic, label_count; 
// 
// #define read_u32be(x, y)                          \
//   assert(fread(&x, 4, 1, y) != 0); x = htobe32(x); 
// 
//   read_u32be(magic, images); assert(magic == 2051);
//   read_u32be(img_count, images);
//   read_u32be(rows, images);
//   read_u32be(columns, images);
// 
//   read_u32be(y_magic, labels); assert(y_magic == 2049);
//   read_u32be(label_count, labels);
// 
//   assert(img_count == label_count);
// 
//   struct mnist* first, *p = &(struct mnist) { 0 };
//   first = p;
// 
//   for (uint32_t i = 0; i < img_count && i < MAX_MNIST; i++)
//     {
//       struct mnist* tmp = malloc(sizeof(struct mnist));
//       tmp->image = malloc(rows * columns);
//       assert(fread(tmp->image, 1, rows * columns, images) != 0); 
//       assert(fread(&tmp->label, 1, 1, labels) != 0);
//       tmp->next = NULL; 
//       tmp->act0 = NULL;
//       tmp->rows = rows;
//       tmp->columns = columns;
//       tmp->act0 = new_mat(1, rows * columns, 0);
// 
//       p->next = tmp;
//       p = p->next; 
//     }
// 
//   fclose(images); fclose(labels);
// 
//   return first->next; 
// } 
// void free_activations(struct mnist* x) 
// {
//   free_mat_struct(x->act1);
//   free_mat_struct(x->act2);
//   free_mat_struct(x->act3);
//   free_mat_struct(x->z1);
//   free_mat_struct(x->z2);
//   free_mat_struct(x->z3);
// }
// void free_mnist(struct mnist* x)
// {
//   free(x->image);
//   free_activations(x);
//   free_mat_struct(x->act0);
//   free(x);
// 
// }

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

// erh
#define softmax_body()                                                  \
  matrix* exps = copy_mat(x);                                           \
  {                                                                     \
    long double max = x->mat[0][0];                                     \
    mat_iterator(x, if (x->mat[i][j] > max) max = x->mat[i][j];);       \
    mat_iterator(exps, exps->mat[i][j] = expl(exps->mat[i][j] - max);); \
  }                                                                     \
  long double sum = sum_mat(exps);

matrix* deriv_softmax(matrix* x)
{ 
  softmax_body();

  mat_iterator(exps,
               exps->mat[i][j] =
               (exps->mat[i][j] / sum) * (1 - exps->mat[i][j] / sum));
  return exps; 
} 

matrix* softmax(matrix* x)
{ 
  softmax_body();

  mat_iterator(exps, exps->mat[i][j] /= sum;);

  return exps; 
} 

#define return_map(z, body)                     \
  matrix* ret = new_mat(z->m, z->n, 0);         \
  mat_iterator(z, body);                        \
  return ret;                                   \

#define ret_map(z, value)                       \
  return_map(z, ret->mat[i][j] = value)

long double sigmoid_function(long double z)
{
  return (expl(-z)) / (powl((expl(-z) + 1), 2.0));
} 

long double deriv_sigmoid_function(long double z)
{
  return 1 / (1 + expl(-z));
} 

matrix* sigmoid(matrix* z)
{
  ret_map(z, sigmoid_function(z->mat[i][j]));
}

matrix* deriv_sigmoid(matrix* z)
{
  ret_map(z, deriv_sigmoid_function(z->mat[i][j]));
}

matrix* relu(matrix* z)
{
  ret_map(z, fmax(0, z->mat[i][j]));
}

void forward_pass(struct network* nn)
{
  for (size_t i = 0; i < nn->amount && nn->l[i].act != NULL; i++) {
    struct layer l = nn->l[i];

    l.z = dot_product(l.wgt, l.a);
    nn->l[i + 1].a = l.act(l.z);
  }
}

// matrix** backward_pass(struct mnist* x, matrix* weights[3])
// {
//   matrix** gradients = malloc(sizeof(struct matrix*) * 3), 
//     *error3, *error2, *error1; 
//     
// 
//   {//computar atualização do peso 3 	
//     error3 = copy_mat(x->act3);
//     matrix* z3s = softmax(x->z3, true), *ohl = one_hot(x->label);
//     mat_iterator(error3, 
//                  error3->mat[i][j] -= ohl->mat[i][j];
//                  error3->mat[i][j] *= 2;
//                  error3->mat[i][j] /= x->act3->m;
//                  error3->mat[i][j] *= z3s->mat[i][j]; 
//                  ); 
//     free_mat_struct(z3s); free_mat_struct(ohl); 
// 
//     gradients[2] = outer(error3, x->act2);
//   } 
// 
//   { //computar atualização do peso 2 	
//     matrix* w3t = copy_mat(W3); transpose(w3t);
//     error2 = dot_product(w3t, error3);
//     matrix* sigtmp = sigmoid_activation(x->z2, true);
//     mat_iterator(error2, error2->mat[i][j] *= sigtmp->mat[i][j];);
//     free_mat_struct(w3t); free_mat_struct(sigtmp);
// 
//     gradients[1] = outer(error2, x->act1);
//   }
// 
//   { //computar atualização do peso 1 	
//     matrix* w2t = copy_mat(W2); transpose(w2t);
//     error1 = dot_product(w2t, error2);
//     matrix* sigtmp = sigmoid_activation(x->z1, true);
//     mat_iterator(error1, error1->mat[i][j] *= sigtmp->mat[i][j];);
//     free_mat_struct(w2t); free_mat_struct(sigtmp);
// 
//     gradients[0] = outer(error1, x->act0);
//   }
//   free_mat_struct(error3); free_mat_struct(error2); free_mat_struct(error1);
// 
//   return gradients;
// }

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

// int compute_acc(struct mnist* x)
// {
//   size_t argmax = 0; 
//   mat_iterator(x->act3, 
//                if (x->act3->mat[i][j] >= x->act3->mat[argmax][0]) 
//                  argmax = i;);
// 
//   if (argmax == x->label)
//     return 1;
//   else
//     return 0;
// }
// void test_acc(struct mnist* x, matrix** weights)
// {
//   unsigned int i = 0, j = 0;
//   for (; x != NULL; x = x->next, j++)
//     { 
//       forward_pass(x, weights);
//       i += compute_acc(x);
//       free_activations(x);
//       //printf("teste: %d\n", j + 1);
//     }
//   printf("forward-pass teste: %.2f%%\n", ((float) i / j) * 100);
// }

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
    // se as matrizes tiverem sido salvas corretamente, o ponteiro da stream vai estar
    // posicionado no inicio de uma nova matriz toda vez que load_matrix retornar
    weights[i] = load_matrix(load);
  } 
  return weights;
}

layer new_layer(size_t size, size_t next_size, matrix* (*act)(matrix*))
{
  return (struct layer)
    {.a = new_mat(size, 1, 0),
     .wgt = next_size == 0 ? NULL : setup_weight(next_size, size, 0.8),
     .z = NULL,
     .b = NULL,
     .size = size,
     .act = act
    };
}

struct network* make_nn(const layout* layout)
{
  struct network* ret = malloc(sizeof(struct network));

  // contar o numero de camadas.
  // um NULL como ativação da output layer marca o fim das camadas.
  // favor não esquecer disso.
  for (ret->amount = 1; layout[ret->amount - 1].act != NULL; ret->amount++);
  ret->l = malloc(sizeof(layer) * ret->amount);

  layer_iterator
    (ret,
     int next_size = layout[i].act == NULL ? 0 : layout[i + 1].size;
     ret->l[i] = new_layer(layout[i].size, next_size,  layout[i].act);
     );

  return ret;
}

void free_layer(struct layer l) {
  free_mat_struct(l.a);
  if (l.wgt != NULL) free_mat_struct(l.wgt);
  if (l.z != NULL) free_mat_struct(l.z);
  if (l.b != NULL) free_mat_struct(l.z);
}

void free_nn(struct network* nn)
{
  layer_iterator(nn, free_layer(nn->l[i]));
  free(nn);
}

// note que os vetores são transpostos para que possam caber em uma tela de terminal.
// pesos por serem grandes não ficam bons de qualquer forma mas eles são transpostos
// tambem para que se mantenha a coesão entre pesos e camadas.
void print_nn(struct network* nn, bool wgt, bool limit)
{
  printf("----\n");
  layer_iterator
    (nn,
     layer l = nn->l[i];

     printf("camada %ld (%ld x %ld):\n", i + 1, l.a->m, l.a->n);
     matrix* foo = p_transpose(l.a);
     print_simple_mat(foo, limit);
     free_mat_struct(foo);

     if (wgt && nn->l[i].wgt != NULL){
       printf("\npeso %ld (%ld x %ld):\n", i + 1, l.wgt->m, l.wgt->n);
       foo = p_transpose(nn->l[i].wgt);
       print_long_mat(foo, limit);
       free_mat_struct(foo);
     }

     if (i != nn->amount - 1){
       printf("\n\tvvv\n");
       printf("\n\n");
     }
     );
  printf("----\n");
  
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

int main(void)
{
  // isso garante que a rede neural retorne os mesmos resultados em todos os treinos
  // dado os mesmos parametros
  srand(1);
  
  printf("c-ann\n");

  const layout nn_layout[] = {{4, &sigmoid},
                              {2, &sigmoid},
                              {1, NULL}};

  struct network* nn = make_nn(nn_layout);

  print_nn(nn, false, false);
  free_nn(nn);

  return 0;
}

