/*
 * rwalk.cuh
 *
 */

 #ifndef RWALK_H_
 #define RWALK_H_
 
 #include <stdio.h>
 #include <assert.h>

 #define MAX_NUM_OF_WALK_PER_NODE 10 

 void TrainGPU_rwalk(int max_walk_length, int num_walks_per_node, unsigned long long random_number);
 void GetResultData_rwalk(int max_walk_length, int num_walks_per_node);
 void initializeGPU_rwalk(int max_walk_length, int num_walks_per_node);
 void cleanUpGPU_rwalk();
 

 extern int64_t * p_scan_list;
 extern int64_t * v_list;
 extern float * w_list;

 extern long int num_of_nodes;
 extern long long int num_of_edges;

 extern int64_t *global_walk;
 
 #endif /* RWALK_H_ */
 