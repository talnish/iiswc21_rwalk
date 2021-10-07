#include "rwalk.cuh"
#include <stdio.h>
#include <assert.h>

int64_t * d_p_scan_list = NULL;
int64_t * d_v_list = NULL;
float * d_w_list = NULL;
int64_t *d_global_walk = NULL;

int tblocksize = 512;
int nblock;

void __global__ device_rwalk(
	int m_walk_length,
	int n_walks_per_node,
	int total_num_nodes, 
	unsigned long long rnumber, 
	int64_t * d_p_scan_list, int64_t * d_v_list, float * d_w_list, int64_t *d_global_walk){
		int64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if(i >= total_num_nodes){
			return;
		}

		long long int w;
	    for(int w_n = 0; w_n < n_walks_per_node; ++w_n) {
			d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + 0] = i;
			// d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + 0] = i;
			float prev_time_stamp = 0;
			int64_t src_node = i;
			int walk_cnt;
			for(walk_cnt = 1; walk_cnt < m_walk_length; ++walk_cnt) {
			  int valid_neighbor_cnt = 0;
			  for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){
				if(d_w_list[w] > prev_time_stamp){
				  valid_neighbor_cnt++;
				  break;
				}
			  }
			  if(valid_neighbor_cnt == 0) {
				break;
			  }
			  float min_bound = d_w_list[d_p_scan_list[src_node]];
			  float max_bound = d_w_list[d_p_scan_list[src_node]];
			  for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){
				if(d_w_list[w] < min_bound)
				  min_bound = d_w_list[w];
				if(d_w_list[w] > max_bound)
				  max_bound = d_w_list[w];
			  }
			  float time_boundary_diff = (max_bound - min_bound);

			  if(time_boundary_diff < 0.0000001){
				for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){ // We randomly pick 1 neighbor, we just pick the first
					if(d_w_list[w] > prev_time_stamp){
						d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
						src_node = d_v_list[w];
						prev_time_stamp = d_w_list[w];
						break;
					}
				}
				continue; 
			  }
			  
			  double exp_summ = 0;            
			  for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){
				if(d_w_list[w] > prev_time_stamp){
				  exp_summ += exp((float)(d_w_list[w]-prev_time_stamp)/time_boundary_diff);
				}
			  }

			  double curCDF = 0, nextCDF = 0;
			  double random_number = rnumber * 1.0 / ULLONG_MAX;
			  rnumber = rnumber * (unsigned long long)25214903917 + 11;   
			  bool fall_through = false;
			  for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){
				if(d_w_list[w] > prev_time_stamp){
					nextCDF += (exp((float)(d_w_list[w]-prev_time_stamp)/time_boundary_diff) * 1.0 / exp_summ);
					if(nextCDF >= random_number && curCDF <= random_number) {
					  d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
					//   d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = d_v_list[w];
					  src_node = d_v_list[w];
					  prev_time_stamp = d_w_list[w];
					  fall_through = true;
					  break;
				  } else {
					  curCDF = nextCDF;
				  }
				}
			  }
			  if(!fall_through){
				for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){ // This line should not be reached anyway (reaching this line means something is wrong). But just for testing, we randomly pick 1 neighbor, we just pick the first
				  if(d_w_list[w] > prev_time_stamp){
					d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
					// d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = d_v_list[w];
					src_node = d_v_list[w];
					prev_time_stamp = d_w_list[w];
					break; 
				  }
				}
			  }
			}
			if (walk_cnt != m_walk_length){	
			  d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = -1;
			}
			
		}
	}

#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		assert(err == cudaSuccess); \
	} \
}
void cleanUpGPU_rwalk(){
	cudaCheck(cudaFree(d_p_scan_list));
	cudaCheck(cudaFree(d_v_list));
	cudaCheck(cudaFree(d_w_list));
	cudaCheck(cudaFree(d_global_walk));
}

void initializeGPU_rwalk(int max_walk_length, int num_walks_per_node){
	// Device query
	int nDevices;
	cudaCheck(cudaGetDeviceCount(&nDevices));
	int device = 0;
	cudaCheck(cudaSetDevice(device));
	cudaDeviceProp prop;
	cudaCheck(cudaGetDeviceProperties(&prop, device));
	tblocksize = prop.maxThreadsPerBlock;
#if defined(DEBUG)
	printf(" Max Threads Per Block %d\n", tblocksize);
#endif
	nblock = (num_of_nodes - 1) / tblocksize + 1;	
	cudaCheck(cudaMalloc((void**) & d_p_scan_list, (num_of_nodes + 1) * sizeof(int64_t)));

	cudaCheck(cudaMalloc((void**)&d_v_list, num_of_edges * sizeof(int64_t)));

	cudaCheck(cudaMalloc((void**) & d_w_list, num_of_edges * sizeof(float)));

	cudaCheck(cudaMalloc((void**) & d_global_walk, num_of_nodes * max_walk_length * MAX_NUM_OF_WALK_PER_NODE * sizeof(int64_t)));

}

void TransferDataToGPU_rwalk(){
	cudaCheck(cudaMemcpy( d_p_scan_list, p_scan_list,
				(num_of_nodes + 1) * sizeof(int64_t) , cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy( d_v_list, v_list,
				num_of_edges * sizeof(int64_t) , cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy( d_w_list, w_list,
				num_of_edges * sizeof(float) , cudaMemcpyHostToDevice));
}

void GetResultData_rwalk(int max_walk_length, int num_walks_per_node, long long int offset){
	cudaCheck(cudaMemcpy(global_walk + offset, d_global_walk, num_of_nodes * max_walk_length * num_walks_per_node * sizeof(int64_t), cudaMemcpyDeviceToHost));
}

void TrainGPU_rwalk(int max_walk_length, int num_walks_per_node, unsigned long long random_number) {
	TransferDataToGPU_rwalk();
	int i;	
	for(i = 0; i < num_walks_per_node / MAX_NUM_OF_WALK_PER_NODE; i++){
		device_rwalk<<<nblock,tblocksize>>>(
			max_walk_length,
			MAX_NUM_OF_WALK_PER_NODE,
			num_of_nodes,
			random_number, 
			d_p_scan_list, d_v_list, d_w_list, d_global_walk);
		GetResultData_rwalk(max_walk_length, MAX_NUM_OF_WALK_PER_NODE, (long long int)num_of_nodes * max_walk_length * MAX_NUM_OF_WALK_PER_NODE * i);
	}

	device_rwalk<<<nblock,tblocksize>>>(
		max_walk_length,
		num_walks_per_node % MAX_NUM_OF_WALK_PER_NODE,
		num_of_nodes,
		random_number, 
		d_p_scan_list, d_v_list, d_w_list, d_global_walk);
	GetResultData_rwalk(max_walk_length, num_walks_per_node % MAX_NUM_OF_WALK_PER_NODE, (long long int)num_of_nodes * max_walk_length * MAX_NUM_OF_WALK_PER_NODE * i);
	
#if defined(DEBUG)
	cudaCheck(cudaGetLastError());
	cudaCheck(cudaDeviceSynchronize());
#endif

}

