
#ifndef matrix_c
#define matrix_c

const size_t sdo = sizeof(long double); // todo: não.

typedef struct matrix {
  long double** mat; // array de ponteiros
  size_t m; // fileiras, i
  size_t n; // colunas, j
} matrix;


// (matrix* a, bool limit, int limit_num, const char* x, const char* y)
#define print_mat(a, limit, limit_num, x, y){               \
  if (a != NULL) {\
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
  }                                                       \
  else printf("null matrix!\n");\
} 

// (matrix* a, bool limit)
#define print_simple_mat(a, limit)\
  print_mat(a, limit, 10, "..%ld..\t%.2LF " , "%.2LF\t")

#define print_long_mat(a, limit)\
  print_mat(a, limit, 3, "..%ld..\t%.10LF\t" , "%.10LF\t")

void print_info_mat(matrix* a)
{
  printf("%ldx%ld\n", a->m, a->n);
}

#define mat_iterator(mat, body)                 \
  for (size_t i = 0; i < mat->m; i++){          \
    for (size_t j = 0; j < mat->n; j++){        \
      body;                                     \
        }                                       \
  } 

#define mat_mat_ops(a, b, sign) \
  mat_iterator(a, a->mat[i][j] = \
      a->mat[i][j] \
      sign \
      b->mat[i][j];\
      ); 

#define mat_scale_ops(a, s, sign) \
  mat_iterator(a, a->mat[i][j] = \
      a->mat[i][j] \
      sign \
      s; \
      ); 

#define mat_sum_mat(a, b)\
  mat_mat_ops(a, b, +);

#define mat_minus_mat(a, b)\
  mat_mat_ops(a, b, -);

#define mat_times_mat(a, b)\
  mat_mat_ops(a, b, *);

#define mat_div_mat(a, b)\
  mat_mat_ops(a, b, /);

#define mat_sum_sca(a, s)\
  mat_scale_ops(a, s, *);

#define mat_minus_sca(a, s)\
  mat_scale_ops(a, s, -);

#define mat_scale(a, s)\
  mat_scale_ops(a, s, *);

#define mat_div_sca(a, s)\
  mat_scale_ops(a, s, /);

#define mat_pow(a, s)\
  mat_iterator(a, a->mat[i][j] = pow(a->mat[i][j], s )); 

// amo C



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
  free(m); 
}

matrix* transpose(matrix* a)
{ 
  long double** retmat = alloc_mat(a->n, a->m, 0);

  mat_iterator(a, retmat[j][i] = a->mat[i][j];);

  // trocar matrizes
  free_mat_arrays(a->mat, a->m); 
  size_t tmp = a->m;
  a->m = a->n;
  a->n = tmp;
  a->mat = retmat;

  return a;
}

matrix* p_transpose(matrix* a)
{ 
  long double** retmat = alloc_mat(a->n, a->m, 0);

  mat_iterator(a, retmat[j][i] = a->mat[i][j];);

  return new_mat_struct(a->n, a->m, retmat);
}

matrix* dot_product(matrix* a, matrix* b)
{ 
  bool transposed = false;

  // print_info_mat(a);
  // print_info_mat(b);
  // tratar vetores como matrizes de coluna caso possivel
  if (a->n == b->n && b->m == 1){ 
    b = p_transpose(b);
    transposed = true;
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
  if (transposed)
    free_mat_struct(b);

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

long double sum_mat(matrix* x)
{
  long double sum = 0;
  mat_iterator(x, sum += x->mat[i][j];);
  return sum; 
}


void dump_matrix(FILE* file, matrix* x)
{
  /* 4 bytes (big-endian) representando o numero de fileiras da matriz (m)
   * 4 bytes (big-endian) representando o numero de colunas da matriz (n)
   * m x n long doubles em ordem representando a matriz
   */
  size_t ret = fwrite(&x->m, sizeof(size_t), 1, file);
  if (ret != 1){ 
    fprintf(stderr, "size_t ret = fwrite(&x->m, sizeof(size_t), 1, file); <---");
    exit(1);
  }
  fwrite(&x->n, sizeof(size_t), 1, file);

  for (size_t i = 0; i < x->m; i++){
    fwrite(x->mat[i], sizeof(long double), x->n, file);
  } 
}

// void dump_params(matrix* weights[3])
// {
//   FILE* dump = fopen("c_ann.weights", "w"); assert(dump != NULL);
// 
//   for (int i = 0 ; i < 3; i++){
//     dump_matrix(dump, weights[i]);
//   }
// 
//   fclose(dump); 
// }

matrix* load_matrix(FILE* load)
{
  size_t m, n; 
  fread(&m, sizeof(size_t), 1, load);
  fread(&n, sizeof(size_t), 1, load);

  matrix* ret = new_mat(m, n, 0); 
  mat_iterator(ret, fread(&ret->mat[i][j], sizeof(long double), 1,  load);); 

  return ret; 
}

// matrix** load_params(FILE* load)
// { 
//   printf("load!\n");
//   matrix** weights = malloc(sizeof(struct matrix*) * 3);
// 
//   for (int i = 0 ; i < 3; i++){
//     // se as matrizes tiverem sido salvas corretamente, o ponteiro da stream vai estar
//     // posicionado no inicio de uma nova matriz toda vez que load_matrix retornar
//     weights[i] = load_matrix(load);
//   } 
//   return weights;
// }

#define return_map(z, body)                     \
  matrix* ret = new_mat(z->m, z->n, 0);         \
  mat_iterator(z, body);                        \
  return ret;                                   \

#define ret_map(z, value)                       \
  return_map(z, ret->mat[i][j] = value)

long double mat_average(matrix* a)
{
  return sum_mat(a) / (a->m * a->n);
}

#endif
