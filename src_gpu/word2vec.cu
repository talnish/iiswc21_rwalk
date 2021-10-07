#include "word2vec.cuh"
#include <stdio.h>
#include <assert.h>

__constant__ real expTable_c[EXP_TABLE_SIZE];

real * d_syn0 = NULL;
real * d_syn1neg = NULL;
int  * d_sen = NULL;
int  * d_sen_length = NULL;
unsigned int * d_random = NULL;
int * d_table = NULL;

int maxThreadsPerBlock = 1024;
int shared_mem_usage;

void __global__ device_memset(real * array, int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		array[idx] = 0;
}


__device__ void reduceInWarp(float * f, int idInWarp){
	#if defined PARALLEL_RED && PARALLEL_RED == 1
		#if defined THREADS_PER_WORD && THREADS_PER_WORD <= 32
			#if defined THREADS_PER_WORD && THREADS_PER_WORD == 32	
			f[idInWarp] += f[idInWarp + 16];
			#endif		
			#if defined THREADS_PER_WORD && THREADS_PER_WORD >= 16	
			f[idInWarp] += f[idInWarp + 8];
			#endif	
			#if defined THREADS_PER_WORD && THREADS_PER_WORD >= 8	
			f[idInWarp] += f[idInWarp + 4];
			#endif	
			#if defined THREADS_PER_WORD && THREADS_PER_WORD >= 4	
			f[idInWarp] += f[idInWarp + 2];
			#endif	
			#if defined THREADS_PER_WORD && THREADS_PER_WORD >= 2	
			f[idInWarp] += f[idInWarp + 1];
			#endif
		#else	
		for (unsigned int i=THREADS_PER_WORD /2; i>32; i>>=1) {
			if (idInWarp < i) {
				f[idInWarp] += f[idInWarp + i];
			}
			__syncthreads();
		}
		if (idInWarp < 32){
			f[idInWarp] += f[idInWarp + 32];
			f[idInWarp] += f[idInWarp + 16];
			f[idInWarp] += f[idInWarp + 8];
			f[idInWarp] += f[idInWarp + 4];
			f[idInWarp] += f[idInWarp + 2];
			f[idInWarp] += f[idInWarp + 1];
		}
		#endif
	#else
		if (idInWarp == 0){
			// float sum = 0;
			for(int i = 1; i < THREADS_PER_WORD; i++){
				f[idInWarp] += f[idInWarp + i];
			}
			// f[0] = sum;
		}
		__syncthreads();
	#endif
}

void __global__ device_cbow(long id, int layer1_size, int layer1_size_aligned,
		int window, int negative, int table_size, int vocab_size,
		int * d_sen, int * d_table,
		volatile float * d_syn0, volatile float *d_syn1neg,
		// float * d_syn0, float *d_syn1neg,
		unsigned int * d_random, int * d_sen_length, float alpha, int syn0_size){

    int batch_id = blockIdx.y;
	int sentence_position = (threadIdx.x / THREADS_PER_WORD) + (blockDim.x / THREADS_PER_WORD) * blockIdx.x;
	int idInWarp = threadIdx.x % THREADS_PER_WORD;

	extern __shared__ float shared[];
	float * f = shared + (threadIdx.x / THREADS_PER_WORD) * THREADS_PER_WORD;
	// float * neu1 = shared + BLOCK_SIZE + (threadIdx.x / THREADS_PER_WORD) * layer1_size_aligned;
	float * neu1e= shared + BLOCK_SIZE + (blockDim.x / THREADS_PER_WORD) * layer1_size_aligned + (threadIdx.x / THREADS_PER_WORD) * layer1_size_aligned;

	if (sentence_position < d_sen_length[batch_id]) {
		unsigned long long next_random = d_random[batch_id * MAX_SENTENCE_LENGTH + sentence_position];

		// for (int sentence_idx = 0; sentence_idx < sentence_num; sentence_idx++){

			for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD) neu1e[c] = 0;

			next_random = next_random * (unsigned long long)25214903917 + 11;
			int b = next_random % window;
			int word = d_sen[batch_id * MAX_SENTENCE_LENGTH + sentence_position];
			// in -> hidden
			// int cw = 0;
			for (int a = b; a < window * 2 + 1 - b; a++){
				if (a != window) {
					int w = sentence_position - window + a;
					if (w < 0)
						continue;
					if (w>= d_sen_length[batch_id])
						continue;
					int last_word = d_sen[batch_id * MAX_SENTENCE_LENGTH + w];
					int l1 = last_word * layer1_size_aligned;
					for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
						// neu1[c] += d_syn0[c + last_word * layer1_size_aligned];
						neu1e[c] = 0;

			// 		cw++;
			// 	}
			
			// if (cw) {
				// for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
				// 	neu1[c] /= cw;
			
			// NEGATIVE SAMPLING
			int target, label;
				for (int d = 0; d < negative + 1; d++) {


					if (d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = d_table[(next_random >> 16) % table_size];
						if (target == 0)
							target = next_random % (vocab_size - 1) + 1;
						if (target == word)
							continue;
						label = 0;
					}
					int l2 = target * layer1_size_aligned;
					f[idInWarp] = 0;
				
					
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD){
						// f[idInWarp] += neu1[c] * d_syn1neg[c + l2];   
						f[idInWarp] += d_syn0[c + l1] * d_syn1neg[c + l2];   
					}
					
					#if defined THREADS_PER_WORD && THREADS_PER_WORD >= 64
					__syncthreads();
					#endif
					// Do reduction here;
					reduceInWarp(f, idInWarp);

					#if defined THREADS_PER_WORD && THREADS_PER_WORD >= 64
					__syncthreads();
					#endif
					
					float g;
					if (f[0] > MAX_EXP)
						g = (label - 1) * alpha;
					else if (f[0] < -MAX_EXP)
						g = (label - 0) * alpha;
					else
						g = (label - expTable_c[(int) ((f[0] + MAX_EXP)
									* (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

					//__syncthreads();	
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						neu1e[c] += g * d_syn1neg[c + l2];
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						// d_syn1neg[c + l2] += g * neu1[c];
						d_syn1neg[c + l2] += g * d_syn0[c + l1];
					
				}
			// hidden -> in
			// for (int a = b; a < window * 2 + 1 - b; a++)
			// 	if (a != window) {
			// 		int w = sentence_position - window + a;
			// 		if (w < 0)
			// 			continue;
			// 		if (w >= MAX_SENTENCE_LENGTH)
			// 			continue;
			// 		int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];

					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						d_syn0[c + l1] += neu1e[c];

				}
			}
			__threadfence_block();
			// }
		// }// End for sentence_idx
		// Update d_random
		if (idInWarp == 0 ) d_random[batch_id * MAX_SENTENCE_LENGTH + sentence_position] = next_random;
	}
}

#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		assert(err == cudaSuccess); \
	} \
}
void cleanUpGPU(){
	cudaCheck(cudaFree(d_syn1neg));
	// cudaCheck(cudaFree(d_avg_syn1neg));
	cudaCheck(cudaFree(d_syn0));
	// cudaCheck(cudaFree(d_avg_syn0));
	cudaCheck(cudaFreeHost(sen));
	cudaCheck(cudaFree(d_sen));
	cudaCheck(cudaFreeHost(sen_length));
	cudaCheck(cudaFree(d_sen_length));
	cudaCheck(cudaFree(d_random));
	cudaCheck(cudaFree(d_table));
}
void initializeGPU(){
	// Device query
	int nDevices;
	cudaCheck(cudaGetDeviceCount(&nDevices));
	int device = 0;
	cudaCheck(cudaSetDevice(device));
	cudaDeviceProp prop;
	cudaCheck(cudaGetDeviceProperties(&prop, device));
	maxThreadsPerBlock = prop.maxThreadsPerBlock;

	real * h_expTable = (real *)malloc((EXP_TABLE_SIZE ) * sizeof(real));
	for (int i = 0; i < EXP_TABLE_SIZE; i++) {
		h_expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
		h_expTable[i] = h_expTable[i] / (h_expTable[i] + 1);
	}
	cudaCheck(cudaMemcpyToSymbol(expTable_c, h_expTable, sizeof(real) * EXP_TABLE_SIZE));
	free(h_expTable);

	if (negative>0) {
		int syn1neg_size = vocab_size * layer1_size_aligned;
		cudaCheck(cudaMalloc((void**) & d_syn1neg, syn1neg_size * sizeof(real)));
		// call memset kernel
		device_memset<<<syn1neg_size / maxThreadsPerBlock + 1, maxThreadsPerBlock>>>(d_syn1neg, syn1neg_size);
		cudaCheck(cudaGetLastError());
		cudaCheck(cudaDeviceSynchronize());

	}

	int syn0_size = vocab_size * layer1_size_aligned;
	
	cudaCheck(cudaMalloc((void**) & d_syn0, syn0_size * sizeof(real)));
	cudaCheck(cudaMemcpy(d_syn0, syn0, syn0_size * sizeof(real), cudaMemcpyHostToDevice));

	cudaCheck(cudaMallocHost((void**)&sen, (MAX_SENTENCE_NUM * BATCH_SYN * MAX_SENTENCE_LENGTH) * sizeof(int) ));
	cudaCheck(cudaMalloc((void**)& d_sen, (BATCH_SYN * MAX_SENTENCE_LENGTH) * sizeof(int) ));

	cudaCheck(cudaMallocHost((void**)&sen_length, (MAX_SENTENCE_NUM * BATCH_SYN) * sizeof(int) ));
	cudaCheck(cudaMalloc((void**)& d_sen_length, (BATCH_SYN) * sizeof(int) ));

	cudaCheck(cudaMalloc((void**) & d_random, BATCH_SYN * MAX_SENTENCE_LENGTH * sizeof(unsigned int)));
	int h_random[BATCH_SYN * MAX_SENTENCE_LENGTH];
	for (int i = 0 ; i < BATCH_SYN * MAX_SENTENCE_LENGTH; i++) h_random[i] = (unsigned int) rand();
	cudaCheck(cudaMemcpy(d_random, h_random, BATCH_SYN * MAX_SENTENCE_LENGTH * sizeof(unsigned int), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**) & d_table, table_size * sizeof(int)));
	cudaMemcpy(d_table, table, table_size * sizeof(int), cudaMemcpyHostToDevice);

	shared_mem_usage = (BLOCK_SIZE + (BLOCK_SIZE/THREADS_PER_WORD) * layer1_size_aligned * 2) * sizeof(real);

}

void TransferDataToGPU(long id){
	cudaCheck(cudaMemcpy( d_sen , sen + (int) id * MAX_SENTENCE_LENGTH * BATCH_SYN,
				(BATCH_SYN * MAX_SENTENCE_LENGTH) * sizeof(int) , cudaMemcpyHostToDevice));

	cudaCheck(cudaMemcpy( d_sen_length , sen_length + (int) id * BATCH_SYN,
				(BATCH_SYN) * sizeof(int) , cudaMemcpyHostToDevice));
}

void GetResultData(){
	cudaCheck(cudaMemcpy(syn0, d_syn0, vocab_size * layer1_size_aligned * sizeof(real), cudaMemcpyDeviceToHost));
}

void TrainGPU(long id, float alpha) {
	TransferDataToGPU(id);
	int syn0_size = vocab_size * layer1_size_aligned;

	dim3 numBlock(MAX_SENTENCE_LENGTH / (BLOCK_SIZE/THREADS_PER_WORD) + 1, BATCH_SYN);
	device_cbow<<<numBlock,BLOCK_SIZE, shared_mem_usage >>>(id, layer1_size, layer1_size_aligned, window,
			 negative, table_size,  vocab_size,	 d_sen, d_table, d_syn0, d_syn1neg, d_random, d_sen_length, alpha, syn0_size);
	
#if defined(DEBUG)
	cudaCheck(cudaGetLastError());
	cudaCheck(cudaDeviceSynchronize());
#endif

}

