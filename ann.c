//
// ffnnc feed forward neural network em C.

// todo:
// vieses
// batch sizes

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

typedef enum activation_func { NONE = 0,  SIGMOID, RELU, STEP, SOFTMAX } activation_func;

// funções de ativação
typedef struct act_function {

    // nome
    char* name;

    // esse enumerador quando passado para act_func_lookup[] retorna uma struct act_function para uso
    // em uma rede neural. util porque isso significa que alguma fonte externa como um arquivo pode
    // dizer ao programa qual função deve ser utilizada.
    activation_func id; 

    // a função em si.
    matrix* (*func)(matrix*); 

} act_function; 

// camadas da rede neural.
typedef struct layer {

    // ativações. resultado de todos os paramatros calculados da camada anterior. a fim de praticidade,
    // isso representa todos os neurônios da camada.
    matrix* a;

    // pesos (weights). parametros. a importancia de cada conexão entre neurônios.
    matrix* wgt;

    // pesos aplicados. salvos para serem usados na hora de calcular os gradients.
    matrix* z;

    // vieses (biases). parametros iguais aos pesos. a única diferença é que são os neuronios e 
    // não as conexões possuem vieses.
    matrix* b;

    // correções dos pesos. calculados após os resultados.
    matrix* error;

    // se machine learning fosse sobre escalar montanhas, um gradient seria a direção pro
    // pico mais alto da montanha.
    matrix* gradient;

    // função de ativação para a proxima camada. pra si própia no caso da ultima camada.
    // ativação é calculada depois da soma de todos os parametros.
    // mesma função que a função de ativação em act_info logo abaixo.
    matrix* (*act)(matrix*); 

    // informações sobre a função de ativação.
    act_function act_info;

    // numero de neurônios.
    size_t size;

} layer;

// rede neural. camadas em serie.
// a primeira camada serve apenas de enfeite. toda vez que nós previsamos fazer o forward
// propagation criamos uma nova camada com todas as ativações correspondentes ao
// formato dos dados e dizemos pra rede neural que aquela será a nova primeira camada.
// são dezenas de inputs diferentes no final das contas.
typedef struct network {

    // todas as camadas.
    layer** l; 

    // coeficiente de apredizado. um hyperparametro. favor manter em quantidades bem 
    // pequenas (0.05 - 0.0001). 
    // ao contrario do que você possa pensar, numeros maiores não necessariamente significam que a 
    // rede neural vá aprender com maior eficiência.
    long double learn_rate;

    // numero de camadas
    size_t nlayers; 

} network;


// facilitar a criação de uma dessas redes neurais. cada layout representa uma camada
// da futura rede neural.
// varios desses em uma array é o jeito que make_nn() entende como uma rede neural deve
// ser criada. 
// caso não queira problemas, a array deve terminar com uma ultima camada com o numero de 
// neuronios sendo 0 e a ativação sendo NONE. 100% das vezes. sendo assim a penultima
// camada especificada é a ultima camada da rede neural.
// exemplo:
// struct layout foo[] = {
//          {20, SIGMOID},  /*input*/
// 			{10, SOFTMAX},  /*segunda camada*/
// 			{5, NONE},      /*output, ultima camada*/
// 			{0, NONE}}	    /*fim do layout*/
typedef struct layout {

    // quantos neurônios por camada.
    int size; 

    // função de ativação
    activation_func act; 

} layout;


// conteiner genérico de dados para treino. não tão genérico quanto eu gostaria.
typedef struct train_data {

    // conteudo. em qualquer formato.
    uint8_t* data;

    // bytes por item.
    size_t size;

    // quantos itens.
    size_t nmemb;

    // resultado esperado.
    int label; 

    // vetor que servirá de input pra camada de input.
    matrix* input; 

    struct train_data* next;

} train_data, test_data;

typedef enum dataset_type { training = 1, testing } dataset_type;

// conteiner com todos os dados para treinamento e mais algumas informações.
typedef struct train_data_context {
    char* dataset_id;
    train_data* train;
    size_t ntrain;
    train_data* test;
    size_t ntest;
    size_t test_freq;
} td_context;

// wip
typedef struct config {
    layout* layout;
} config;

#define layer_iterator(nn, body)                \
    for (size_t l = 0; l < nn->nlayers; l++){     \
        body;                                       \
    } 

#define free_act(nn, x)                         \
    if (nn->l[l]->x != NULL){                     \
        free_mat_struct(nn->l[l]->x);               \
        nn->l[l]->x = NULL;                         \
    }

matrix* normalize_pixels(train_data* x)
{
    matrix* ret = new_mat(1, x->nmemb, 0);

    for (size_t i = 0; i < ret->n; i++)
        ret->mat[0][i] = (long double) x->data[i] / 255; 

    return ret;
} 

// importar dataset MNIST. um conjunto de caracteres desenhados para o aprendizado de uma rede
// neural capaz de reconhecer digitos escritos em uma (pequena, 28x28) imagem. 
// http://yann.lecun.com/exdb/mnist/
void parse_idx_files(char* x, char* y, td_context* tdc, int ds_type)
{
    tdc->dataset_id = "MNIST";

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

    size_t max;

    if (ds_type == training)
        max = tdc->ntrain;
    else
        max = tdc->ntest;
    for (size_t i = 0; i < img_count && i < max; i++)
    {
        printf("\rimportantdo MNIST %ld/%ld...", i + 1, max);
        fflush(stdout);
        train_data* tmp = malloc(sizeof(struct train_data));
        if (tmp == NULL){
            fprintf(stderr, "train_data* tmp = malloc(sizeof(struct train_data)) <---\n"); 
            exit(1);
        }

        tmp->data = malloc(rows * columns);
        if (tmp->data == NULL){
            fprintf(stderr, "tmp->data = malloc(rows * columns) <---"); 
            exit(1);
        }
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
        tmp->size = 1;

        // normalizar pixeis das imagens para que todos fiquem com valores entre
        // 0 e 1. com 0 sendo preto e 1 sendo branco.
        tmp->input = normalize_pixels(tmp);

        // não seria melhor ler as imagens dentro de normalize?
        free(tmp->data);

        p->next = tmp;
        p = p->next; 
    }

    fclose(images); fclose(labels);
    printf("\r\rMNIST importado com sucesso!\n");
    fflush(stdout);

    if (ds_type == training)
        tdc->train = first->next; 
    else 
        tdc->test = first->next; 
} 

train_data** select_node(struct train_data** head, int i) {
    struct train_data** node = head;
    for (int k = 0; k < i; k++) {
        node = &((*node)->next);
    }
    return node;
}

void swap_td(train_data* a, train_data* b) {
    void* temp_data = a->data;
    int temp_label = a->label;
}

void shuffle_train_data(td_context* td) {
    
    train_data** head = &td->train;

    // count the number of nodes in the linked list
    train_data* last = *head;
    while (last != NULL)  last = last->next; 
    
    // perform the Fisher-Yates shuffle algorithm
    for (int i =  - 1; i > 0; i--) {
        // generate a random index between 0 and i
        int j = rand() % (i + 1);
        
        // swap the ith and jth nodes in the linked list
        train_data* node_i = *head;
        train_data* node_j = *head;
        for (int k = 0; k < i; k++) {
            node_i = node_i->next;
        }
        for (int k = 0; k < j; k++) {
            node_j = node_j->next;
        }

        swap_td(node_i, node_j);
    }
}



// void shuffle_train_data(td_context* td) {
// 
//     train_data* last = td->train;
// 
//     while (last->next != NULL) last = last->next;
// 
//     for (size_t i = td->ntrain - 1; i > 0; i--) {
//         size_t j = rand() % (i + 1);
// 
//         train_data* node_i = td->train;
//         train_data* node_j = td->train;
// 
//         for (size_t k = 0; k < i; k++) node_i = node_i->next;
//         for (size_t k = 0; k < j; k++) node_j = node_j->next;
// 
//         train_data temp = *node_i;
//         train_data* temp_next = node_i->next;
//         *node_i = *node_j;
//         node_i->next = node_j->next;
//         *node_j = temp;
//         node_j->next = temp_next;
// 
//     }
// 
// }

matrix* one_hot(uint8_t x)
{
    assert(x < 10);

    matrix* ret = new_mat(1, 10, 0);

    ret->mat[0][x] = 1;

    return ret;
} 

// terrivel
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

matrix* sigmoid(matrix* z)
{
    return_map(z, 
            long double zij = z->mat[i][j];
            ret->mat[i][j] = expl(-zij) / powl(expl(-zij) + 1, 2.0)
            );
}

matrix* deriv_sigmoid(matrix* z)
{
    return_map(z, 
            long double zij = z->mat[i][j];
            ret->mat[i][j] = 1 / (1 + expl(-zij))
            );
}

matrix* relu(matrix* z)
{
    ret_map(z, fmax(0, z->mat[i][j]));
}

#define threshold 0
// basicamente faz com que neurônios virem perceptrons
matrix* step(matrix* z)
{
    ret_map(z, z->mat[i][j] > threshold ? 1 : 0);
}

void calculate_gradients(network* nn, matrix* target)
{
#define nnlast nn->nlayers - 1

    matrix* out_error = copy_mat(nn->l[nnlast]->a);

    mat_minus_mat(out_error, target); 
    nn->l[nnlast]->error = out_error;
    matrix* l_1 = out_error;

    // aqui o bixo pega
    for (int l = nnlast - 1; l > 0; l--)
    { 
        matrix* i_error = copy_mat(nn->l[l]->a); 
        matrix* dot = dot_product(nn->l[l + 1]->error, nn->l[l]->wgt);

        mat_iterator(i_error,
                i_error->mat[i][j] = 
                nn->l[l]->a->mat[i][j] * (1 - nn->l[l]->a->mat[i][j]) * dot->mat[i][j]; 
                );
        free_mat_struct(dot);

        nn->l[l]->error = i_error;
        nn->l[l]->gradient = outer(nn->l[l + 1]->error, nn->l[l]->a);
        l_1 = i_error; 
    } 
} 

void update_parameters(network* nn)
{
    layer_iterator(nn,
            struct layer* lay = nn->l[l];
            if (lay->gradient == NULL) continue;

            /* long double alpha = nn->learn_rate * mat_average(lay->gradient); */
            /* mat_iterator(lay->wgt, lay->wgt->mat[i][j] += alpha); */
            mat_iterator(lay->wgt,
                lay->wgt->mat[i][j] -=
                nn->learn_rate * lay->gradient->mat[i][j]);
            free_act(nn, error); free_act(nn, gradient);
            );
}

long double rand_ld(long double max)
{
    return ((long double) rand() / (long double) max) * 0.0245e-8;
}

matrix* setup_weight(size_t m, size_t n, long double k)
{
    matrix* x = new_mat(m, n, 0);

    mat_iterator(x, x->mat[i][j] = rand_ld(k) - rand_ld(k);); 

    return x;
}

struct act_function act_func_lookup[] = { 
    {"none", NONE, NULL},
    {"sigmoid", SIGMOID, &sigmoid},
    {"relu", RELU, &relu},
    {"step", STEP, &step},
    {"softmax", SOFTMAX, &softmax}
}; 

layer* new_layer(size_t size, size_t next_size, activation_func act)
{
    layer* ret = malloc(sizeof(struct layer));

    ret->wgt = next_size == 0 ? NULL : setup_weight(next_size, size, 0.8);
    ret->a = NULL;
    ret->z = NULL;
    ret->b = NULL;
    ret->error = NULL;
    ret->gradient = NULL;
    ret->size = size;
    ret->act = act_func_lookup[act].func;
    ret->act_info = act_func_lookup[act];

    return ret;
}

struct network* make_nn(const layout* layout, long double learn_rate)
{
    // isso garante que a rede neural retorne os mesmos resultados em todos os treinos
    // dado os mesmos parametros
    srand(1);

    struct network* ret = malloc(sizeof(struct network));

    // contar o numero de camadas.
    for (ret->nlayers = 0; layout[ret->nlayers].size != 0; ret->nlayers++);

    ret->l = malloc(sizeof(struct layer*) * ret->nlayers);
    ret->learn_rate = learn_rate;

    layer_iterator(ret,
            // fixme
            int next_size = l + 1 == ret->nlayers ? 0 : layout[l + 1].size;
            ret->l[l] = new_layer(layout[l].size, next_size,  layout[l].act);
            );

    return ret;
}

void free_layer(struct layer* l) {
    if (l->a != NULL) free_mat_struct(l->a);
    if (l->wgt != NULL) free_mat_struct(l->wgt);
    if (l->z != NULL) free_mat_struct(l->z);
    if (l->b != NULL) free_mat_struct(l->z);
    if (l->error != NULL) free_mat_struct(l->error);
    if (l->gradient != NULL) free_mat_struct(l->gradient);
}

void free_nn(struct network* nn)
{
    layer_iterator(nn, free_layer(nn->l[l]));
    free(nn);
}

// note que os vetores são transpostos para que possam caber em uma tela de terminal.
// pesos por serem grandes não ficam bons de qualquer forma mas eles são transpostos
// tambem para que se mantenha a coesão entre pesos e camadas.
void print_nn(struct network* nn, bool wgt, bool limit)
{
    printf("----\n");
    layer_iterator (nn,
            layer* la = nn->l[l];
            matrix* foo = NULL;

            if (la->a != NULL){ 
            printf("camada %ld (%ld x %ld)\n", l + 1, la->a->m, la->a->n);
            foo = p_transpose(la->a);
            print_long_mat(foo, limit);
            free_mat_struct(foo);
            }

            if (wgt && la->wgt != NULL){
            printf("\npeso %ld (%ld x %ld):\n", l + 1, la->wgt->m, la->wgt->n);
            foo = p_transpose(nn->l[l]->wgt);
            print_long_mat(foo, limit);
            free_mat_struct(foo);
            }

            if (l != nn->nlayers - 1){
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

    for (size_t i = 0; i + 1 < nn->nlayers; i++) 
    {
        struct layer* l = nn->l[i]; 
        l->z = transpose(dot_product(l->wgt, l->a)); 
        // print_info_mat(l->a);
        // print_info_mat(l->z);

        if (l->act != NULL){ 
            nn->l[i + 1]->a = l->act(l->z); }
        else nn->l[i + 1]->a = l->z;
    } 

    // aplicar a ativação da camada de output em si mesmo
    if (out_l->act != NULL)
        out_l->a = out_l->act(out_l->a);

    // clear_nn() tentaria desalocar o input caso o contrario
    nn->l[0]->a = NULL;
}

// desaloca todas as ativações mas mantem pesos e vieses 
void clear_nn(network* nn)
{
    layer_iterator(nn, free_act(nn, a); free_act(nn, z));
}

// quanto menor melhor
long double cost_function(matrix* pred, matrix* target)
{
    long double sum = 0;
    mat_iterator(pred, sum = pow(target->mat[i][j] - pred->mat[i][j], 2));
    return sum;
}

long double test_nn(network* nn, td_context* tdc)
{
    train_data* td = tdc->test;
    long double cost = 0, steps = 0;
    for (; td != NULL; td = td->next, steps++)
    {
        printf("\rtestando... %.0LF/%ld", steps, tdc->ntest);
        fflush(stdout);
        forward_pass(td->input, nn);
        matrix* target = one_hot(td->label);
        cost += cost_function(nn->l[nn->nlayers - 1]->a, target );
        // printf("\r\n%.0LF -> %.0LF\n", onehot_argmax(target), onehot_argmax(nn->l[nn->nlayers - 1]->a));
        clear_nn(nn);
        free_mat_struct(target);
    } 
    return cost / steps;
}


void train_nn(network* nn, td_context* tdc, int epochs)
{
    train_data* td = tdc->train;
    train_data* first = td;
    for (int i = 0; i < epochs; i++){ 
        td = first;
        long double cost = 0, steps = 0;
        for (; td != NULL; td = td->next, steps++)
        {
            printf("\rtreinando... %.0LF/%ld (epoch %d)", steps, tdc->ntrain, i + 1);
            fflush(stdout);
            forward_pass(td->input, nn);
            matrix* target = one_hot(td->label);
            cost += cost_function(nn->l[nn->nlayers - 1]->a, target);
            calculate_gradients(nn, target);
            update_parameters(nn);
            clear_nn(nn);
            free_mat_struct(target);
        } 
        printf("\rcost function: %LF (epoch %d)\n", cost / tdc->ntrain, i + 1);

        if (tdc->ntest > 0 && (i + 1) % tdc->test_freq == 0) {
            long double test_cost = test_nn(nn, tdc);
            printf("\rcost function (teste): %LF (%ld epoch(s))\n", test_cost, tdc->test_freq);
        }
    }
}

void print_layout(network* nn) 
{
    layer_iterator(nn, 
            printf("%ld ", nn->l[l]->size);
            if (l + 1 < nn->nlayers)
            printf("> %s > ", nn->l[l]->act_info.name));
    printf("\n"); 
}

void dump_layer(FILE* file, layer* lay)
{ 
    fwrite(&lay->size, sizeof(size_t), 1, file);
    fwrite(&lay->act_info.id, sizeof(activation_func), 1, file);
}

void dump_params(FILE* file, layer* lay)
{
    uint8_t yay = 1, nay = 0;

    if (lay->wgt != NULL) { 
        fwrite(&yay, sizeof(uint8_t), 1, file);
        dump_matrix(file, lay->wgt);

    } else
        fwrite(&nay, sizeof(uint8_t), 1, file); 
}

// apenas pesos por enquanto
matrix* load_params(FILE* file, layer* lay)
{ 
    uint8_t yaynay; 
    fread(&yaynay, sizeof(uint8_t), 1, file);
    if (yaynay > 1){
        fprintf(stderr, "fread(&yaynay, sizeof(uint8_t), 1, file); <---\n");
        exit(1);
    }
    if (yaynay == 1){ 
        return load_matrix(file);
    }
    else return NULL; 
}

/* string de 5 bytes com os caracteres "ffnnc". não contendo \n no final.
 * unsigned integer (4 bytes) representando o numero de camadas.
 * long double (10 bytes) representando o learning rate.
 * para cada camada:
 * 	unsigned integer (4 bytes) com o numero de neuronios da camada.
 * 	unsigned integer (4 bytes) com o id da funçá̃o de ativação da camada.
 * em seguida vem os parametros. por enquanto apenas pesos.
 * para cada camada:
 */
void export_model(network* nn, const char* filename)
{
    FILE* file = fopen(filename, "w"); 
    // magic
    fwrite(&"ffnnc", sizeof(char), 5, file);

    fwrite(&nn->nlayers, sizeof(size_t), 1, file);

    fwrite(&nn->learn_rate, sizeof(long double), 1, file);

    layer_iterator(nn, dump_layer(file, nn->l[l]));

    layer_iterator(nn, dump_params(file, nn->l[l]));

    fclose(file); 
}

network* import_model(char* filename)
{
    FILE* file = fopen(filename, "r");

    char* magic = malloc(sizeof(char) * 6); magic[5] = '\n';
    fread(magic, sizeof(char), 5, file);
    if (!strcmp(magic, "ffnnc")){ 
        fprintf(stderr, "arquivo não é do tipo ffncc!. saindo.\n");
        exit(1);
        return NULL;
    }
    size_t nlayers = 0; 
    fread(&nlayers, sizeof(size_t), 1, file);
    if (nlayers == 0){
        fprintf(stderr, "fread(&nlayers, sizeof(size_t), 1, file); <---\n");
        exit(1);
        return NULL;
    }
    long double learn_rate = 0; 
    fread(&learn_rate, sizeof(long double), 1, file);
    if (learn_rate == 0 || isnan(learn_rate)){
        fprintf(stderr, "fread(&learn_rate, sizeof(long double), 1, file); <---\n");
        exit(1);
        return NULL;
    }

    layout* nn_layout = malloc(sizeof(struct layout) * (nlayers + 1));

    for (size_t i = 0; i < nlayers; i++){ 
        fread(&nn_layout[i].size, sizeof(size_t), 1, file);
        fread(&nn_layout[i].act, sizeof(activation_func), 1, file);
    }
    nn_layout[nlayers].size = 0; nn_layout[nlayers].act = NONE;

    network* nn = make_nn(nn_layout, learn_rate); 

    layer_iterator(nn, free_act(nn, wgt); nn->l[l]->wgt = load_params(file, nn->l[l]));

    fclose(file); 

    return nn;
}

