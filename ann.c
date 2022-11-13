
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <endian.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include "matrix.c"

#define TRAIN_IMAGES_PATH \
"/home/xi/Desktop/prog/lisp/common lisp/ann/data/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH \
  "/home/xi/Desktop/prog/lisp/common lisp/ann/data/train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH \
  "/home/xi/Desktop/prog/lisp/common lisp/ann/data/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH \
  "/home/xi/Desktop/prog/lisp/common lisp/ann/data/t10k-labels.idx1-ubyte"

typedef struct layer {
  // ativações. resultado de todos os paramtros calculados da camada anterior.
  struct matrix* a;
  // pesos (weights). parametros. a importancia de cada conexão entre neurônios.
  // a magia acontece aqui.
  struct matrix* wgt;
  // pesos aplicados. salvos para serem usados na hora de calcular o gradient descent.
  struct matrix* z;
  // vieses (biases). parametros iguais aos pesos. a única diferença é que neuronios
  // tem vieses e não as conexões.
  struct matrix* b;
  size_t size;
  // função de ativação para a proxuma camada. ativação é calculada depois da soma
  // de todos os parametros.
  matrix* (*act)(matrix*);
} layer;

// struct network, de "neural network". camadas em serie
// a primeira camada serve apenas de enfeite. toda vez que nós previsamos fazer um forward
// propagation criamos uma nova camada com todas as ativações correspondentes ao
// formato dos dados e dizemos pra rede neural que aquela será a nova primeira camada.
// são dezenas de dados diferentes que passam por essas camadas no final das contas.
typedef struct network {
  struct layer** l; // array de pointeros pras camadas em si.
  size_t nlayers; // numero de camadas
} network;

// facilitar a criação de uma dessas redes neurais. cada layout representa uma camada
// da futura rede neural.
// varios desses em uma array é o jeito que make_nn() entende como uma rede neural deve
// ser criada. 
// caso não queira problemas a array deve terminar com uma ultima camada com o numero de 
// neuronios sendo 0 e a ativação sendo NULL. 100% das vezes. sendo assim a penultima
// camada especificada é a ultima camada da rede neural.
typedef struct layout {
  int size; // quantos neurônios por camada.
  matrix* (*act)(matrix*); // função de ativação pra proxima camada.
  // pra si própia no caso da ultima camada.
} layout;
// exemplo:
// struct layout foo[] = {{20, &sigmoid}, /*input*/
// 			 {10, &sigmoid},  /*camada 2*/
// 			 {5, %softmax},   /*output*/
// 			 {0, NULL}}; 	  /*fim do layout*/


typedef struct train_data {
  uint8_t* data;
  size_t size;
  size_t nmemb;
  int label;

  matrix* input;

  struct train_data* next;
} train_data, test_data;

// wip
typedef struct config {
  struct layout* layout;
} config;

const long double L_RATE = 0.0005; //não abusar dessa variavel.

// amo C

#define layer_iterator(nn, body)                \
  for (size_t i = 0; i < nn->nlayers; i++){     \
    body;                                       \
  } 

matrix* normalize_pixels(train_data* x)
{
  matrix* ret = new_mat(x->nmemb, 1, 0);

  for (size_t i = 0; i < ret->m; i++)
    ret->mat[i][0] = (long double) x->data[i] / 255; 

  return ret;
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


//http://yann.lecun.com/exdb/mnist/
train_data* parse_idx_files(char* x, char* y, size_t max)
{
  FILE* images = fopen(x, "r"); assert(images != NULL);
  FILE* labels = fopen(y, "r"); assert(labels != NULL);
  uint32_t magic, img_count, rows, columns,
    y_magic, label_count; 

#define read_u32be(x, y)                          \
  assert(fread(&x, 4, 1, y) != 0); x = htobe32(x); 

  read_u32be(magic, images); assert(magic == 2051);
  read_u32be(img_count, images);
  read_u32be(rows, images);
  read_u32be(columns, images);

  read_u32be(y_magic, labels); assert(y_magic == 2049);
  read_u32be(label_count, labels);

  assert(img_count == label_count);

  train_data* first, *p = &(struct train_data) { 0 };
  first = p;

  for (uint32_t i = 0; i < img_count && i < max; i++)
    {
      train_data* tmp = malloc(sizeof(struct train_data));

      tmp->data = malloc(rows * columns);
      if (fread(tmp->data, 1, rows * columns, images) == 0){
        fprintf(stderr, "fread(tmp->data, 1, rows * columns, images) <--\n");
        exit(1);
      }
      if (fread(&tmp->label, 1, 1, labels) == 0) {
        fprintf(stderr, "fread(&tmp->label, 1, 1, labels) <--\n");
        exit(1);
      }
      tmp->next = NULL; 
      tmp->nmemb = rows * columns;
      tmp->size = 8;

      // normalizar pixeis das imagens para que todos fiquem com valores entre
      // 0 e 1. com 0 sendo preto e 1 sendo branco.
      tmp->input = normalize_pixels(tmp);

      // não seria melhor ler as imagens dentro de normalize?
      free(tmp->data);

      p->next = tmp;
      p = p->next; 
    }

  fclose(images); fclose(labels);

  return first->next; 
} 

matrix* one_hot(uint8_t x)
{
  assert(x < 10);

  matrix* ret = new_mat(10, 1, 0);

  ret->mat[x][0] = 1;

  return ret;
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

layer* new_layer(size_t size, size_t next_size, matrix* (*act)(matrix*))
{
  layer* ret = malloc(sizeof(struct layer));

  ret->wgt = next_size == 0 ? NULL : setup_weight(next_size, size, 0.8);
  ret->a = NULL;
  ret->z = NULL;
  ret->b = NULL;
  ret->size = size;
  ret->act = act;

  return ret;
}

struct network* make_nn(const layout* layout)
{
  // isso garante que a rede neural retorne os mesmos resultados em todos os treinos
  // dado os mesmos parametros
  srand(1);

  struct network* ret = malloc(sizeof(struct network));

  // contar o numero de camadas.
  for (ret->nlayers = 0; layout[ret->nlayers].size != 0; ret->nlayers++);

  ret->l = malloc(sizeof(struct layer*) * ret->nlayers);

  layer_iterator
    (ret,
     int next_size = i + 1 == ret->nlayers ? 0 : layout[i + 1].size;
     ret->l[i] = new_layer(layout[i].size, next_size,  layout[i].act);
     );

  return ret;
}

void free_layer(struct layer* l) {
  if (l->a != NULL) free_mat_struct(l->a);
  if (l->wgt != NULL) free_mat_struct(l->wgt);
  if (l->z != NULL) free_mat_struct(l->z);
  if (l->b != NULL) free_mat_struct(l->z);
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
     layer* l = nn->l[i];

     printf("camada %ld (%ld x %ld):\n", i + 1, l->a->m, l->a->n);
     matrix* foo = p_transpose(l->a);
     print_long_mat(foo, limit);
     free_mat_struct(foo);

     if (wgt && nn->l[i]->wgt != NULL){
       printf("\npeso %ld (%ld x %ld):\n", i + 1, l->wgt->m, l->wgt->n);
       foo = p_transpose(nn->l[i]->wgt);
       print_long_mat(foo, limit);
       free_mat_struct(foo);
     }

     if (i != nn->nlayers - 1){
       printf("\n\tvvv\n");
       printf("\n\n");
     }
     );
  printf("----\n");
  
}

void forward_pass(matrix* input, struct network* nn)
{
   layer* out_l = nn->l[nn->nlayers - 1];
  nn->l[0]->a = input;

  for (size_t i = 0; i + 1 < nn->nlayers; i++) {
    struct layer* l = nn->l[i];
    l->z = dot_product(l->wgt, l->a);

    if (l->act != NULL)
      nn->l[i + 1]->a = l->act(l->z);
  }

  // aplicar a ativação da camada de output no resultado da penultima camada
  if (out_l->act != NULL)
    out_l->a = out_l->act(out_l->a);
}

// desaloca todas as ativações mas mantem pesos e vieses 
void clear_nn(network* nn)
{
  layer_iterator(nn,
                 if (nn->l[i]->a != NULL)
                   free_mat_struct(nn->l[i]->a);
                 if (nn->l[i]->z != NULL)
                   free_mat_struct(nn->l[i]->z);
                 );
}

// quanto menor melhor
long double cost_function(matrix* pred, matrix* target)
{
  long double sum = 0;
  mat_iterator(pred,
               sum += pow(pred->mat[i][j] - target->mat[i][j], 2);
               );
  return sum;
}

void train_nn(network* nn, train_data* td)
{
  long double cost = 0;
  long double steps = 0;
  for (; td != NULL; td = td->next, steps++){
    clear_nn(nn);
    forward_pass(td->input, nn);
    cost += cost_function(nn->l[nn->nlayers - 1]->a, one_hot(td->label));
    printf("%d\n", (int) td->label);
  }

  printf("cost function: %LF\n", cost / steps);
}

int main(void)
{
  printf("c-ann!\n");

  const layout nn_layout[] = {{784, &sigmoid},
                              {128, &sigmoid},
                              {64, &sigmoid},
                              {10, NULL},
                              {0, NULL}};

  struct network* nn = make_nn(nn_layout);

  train_data* foo = parse_idx_files(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, 2000);

  train_nn(nn, foo);

  free_nn(nn);

  return 0;
}

