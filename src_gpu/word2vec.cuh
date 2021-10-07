/*
 * word2vec.cuh
 *
 */

 #ifndef WORD2VEC_H_
 #define WORD2VEC_H_
 
 #define MAX_STRING 100
 #define EXP_TABLE_SIZE 1000
 #define MAX_EXP 6
 #define MAX_SENTENCE_LENGTH 64
 #define MAX_CODE_LENGTH 40
 #define MAX_SENTENCE_NUM 16
 #define ALIGNMENT_FACTOR 1 //32
 #define THREADS_PER_WORD 8
 #define BLOCK_SIZE 128
 
 #define BATCH_SYN 16384

 #define PARALLEL_RED 1


 typedef float real;
 
 #include <stdio.h>
 #include <assert.h>
 
 void TrainGPU(long id, float alpha);
 void GetResultData();
 void initializeGPU();
 void cleanUpGPU();
 
extern real *syn0;
extern int * table;
extern long long vocab_size, layer1_size , layer1_size_aligned;
extern int negative , window;
extern int table_size;

extern int * sen;
extern int * sen_length;
 
 #endif /* WORD2VEC_H_ */
 