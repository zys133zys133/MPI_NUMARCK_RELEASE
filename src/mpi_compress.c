#include "stdlib.h"
#include "stdio.h"
#include "mpi.h"
#include "string.h"
#include "heap.h"
#include "assert.h"
#ifdef _ADD_PNETCDF_
#include "pnetcdf.h"
#endif
#include "memory_bit_buffer.h"
#include "zlib.h"
#include "sys/time.h"
#include "sys/stat.h"
#ifdef _ADD_HDF5_
#include "hdf5.h"
#endif
#include "math.h"


#define BLOCK_LOW(id, p, n) ((id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1, p, n)-1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1)
#define BLOCK_OWNER(index, p, n) (((P) * ((index) +1)-1)/(n))
#define ERR(e) {printf("Error: %s\n", nc_strerror(e));}
#define	diff_double(a,b) ((a)>(b)?((a)-(b)):((b)-(a)))


static int DEBUG;
static int WRITE_FLAG;

/*
 * meta data for block
 */

struct block_element
{
	int start_index;
	int data_num;
	int uncompress_data_point_num;
	int previous_data_point_num;
	long uncom_buff_length;
	long com_buff_length;
	unsigned char *uncom_buff;
	unsigned char *com_buff;
};

/*
 * helper data structure for a block, used in index transfer among
 * neighbhors
 */

struct g_block_map_element
{
	long unit_num;
	long start_index;
	long end_index;
	long real_start_index;
	long real_end_index;
	long real_unit_num;
	int send_r_id;
	long send_r_num;
	int send_l_id;
	long send_l_num;
};

//int zlib_compression_level = Z_BEST_SPEED;
//int zlib_compression_level = Z_BEST_COMPRESSION;
int zlib_compression_level = Z_DEFAULT_COMPRESSION;

//default number of max bins for top-k
int max_predict_bin = 10000000;

int output_format = 0;	//	0:pnetcdf	1:hdf5
int input_format = 0;	//	0:pnetcdf	1:hdf5

char *var_name = NULL;	//variable name to be compressed in input file
int var_index;		//variable index in input file

int float_flag = 0;	//	0:double data type	1:float data type

//range of max & min change ratio considered in top-k
double max_change_ratio = 40.0;
double min_change_ratio = -40.0;

int helper_array[8][6];		//bits operation helper array

int compression_method = 2;		//0:kmeans	1:equal-bin	2:simple-grouping	3:log-bin
int index_table_com_flag = 1;		//0:do not compress index table	1:compress index table using zlib
int cluster_table_reuse = 0;		//1:reuse previous cluster center table if it reduce space	(incompatiable with NUMARCK decompression)
int cluster_table_can_reuse;		//helper variable

//timers
long total_total_time = 0;
long total_change_ratio_time = 0;
long total_binning_time = 0;
//long total_indexing_time = 0;
long total_assign_index_time = 0;
long total_trans_index_time = 0;
long total_bits_opt_time = 0;
long total_zlib_time = 0;
long total_predict_time = 0;
long total_write_file_time = 0;
long init_bin_time;
long local_bit_runtime;
long local_zlib_runtime;
long top_k_selection_time;
long bits_shift_time;
long phase_1_total = 0;
long phase_2_total = 0;

long long total_raw_file_size = 0;		//size of compressed NUMARCK file
long long last_file_size = 0;		//used for multi-variable compression

int var_num = 1;			//number of var to be compressed

int dim0_block_size = 165;
int dim_nxb = 32;
int dim_nyb = 32;

int block_size;				//number of elements in a block

int block_length = 1024;		//block length (Bytes)

char *input_file_1 = NULL;
char *input_file_2 = NULL;
char *output_file = NULL;

int rank,size;				//MPI rank & size
int data_array_row_num;			//number of elements in this computational node
int class_num;				//number of bins selected by NUMARCK
int uncompress_count;			//number of incompressible element in this node
int block_array_size;			//number of blocks in this node
int g_block_num;
int zero_change_ratio_num;
int t_change_ratio_array_length;
int *membership,*t_membership;
//int *change_ratio_flag_array;
double *data_array1;
double *data_array2;
double *change_ratio_array;
double *t_change_ratio_array;
double *cluster_centroids;
double *uncompress_table;
long double incompressible_ratio;
long double ave_ratio_error;
long double zlib_compression_ratio;

struct g_block_map_element *g_block_map;
struct block_element *block_array;

struct timeval read_file_start,numarck_start,end,end1;
struct timeval phase_1_start,phase_1_end,phase_2_start,phase_2_end,phase_3_start,phase_3_end;
struct timeval change_ratio_calculation_start,indexing_start,prediction_start,index_table_compression_start,write_file_start,write_file_end,indexing_add_start,indexing_add_end,write_file_phase_1,binning_start,binning_end,trans_index_start,bits_opt_start;
struct timeval phase_1_time,phase_2_time;

extern double** mpi_read(int       isBinaryFile,  /* flag: 0 or 1 */
                 char     *filename,      /* input file name */
                 int      *numObjs,       /* no. data objects (local) */
                 int      *numCoords,     /* no. coordinates */
                 MPI_Comm  comm);
extern int mpi_kmeans(double    *objects,     /* in: [numObjs][numCoords] */
               int        numCoords,   /* no. coordinates */
               int        numObjs,     /* no. objects */
               int        numClusters, /* no. clusters */
               double      threshold,   /* % objects change membership */
               int       *membership,  /* out: [numObjs] */
               double    *clusters,    /* out: [numClusters][numCoords] */
               MPI_Comm   comm);
void read_cmip_file(int index);
void read_flash_file(int index);
void read_flash_file_all(int index);
void read_cmip_file_all(int index);
void read_cisl_file(int index);
void read_stir_file(int index);
void itoa(char *ch,int num);
void calc_change_ratio(double *E);
//double diff_double(double a,double b);
void binning(int *B, double *E);
static void handle_error(int status, int lineno);
void mpi_equal_bin(double *t_change_ratio_array,int index,int r_class_num,int *t_membership,int *B,double *E);
void mpi_log_bin(double *t_change_ratio_array,int index,int r_class_num,int *t_membership,int *B, double *E);
void load_netcdf_file_all_float(char *path,double **array);
void load_netcdf_file_all_double(char *path,double **array);
void update_cluster_centroids_and_membership(double *E);
void index_data();
void indexing();
void compress_block(int block_index);
int calc_mini_block_num(int start,int length,int dim0_size);
int calc_send_num(int b_index,int s_index,int e_index);
int left_size(int index);
int right_size(int index);
int block_index(int index);
void data_split();
void cluster_initial_centroids_collection(int index,double *t_change_ratio_array);
void mpi_simple_grouping2(double *t_change_ratio_array,int index,int r_class_num,int *t_membership,int zero_change_ratio_num,int *B, double *E);
void load_cmip(char *path,int index,double **array1,double **array2);
int simple_grouping_prediction(struct t_center_element *t_center,int total_num,int total_bin_num,int already_compressed_num);
long int time_diff(struct timeval t_s,struct timeval t_e);
void pnetcdf_write();
void hdf5_write();
unsigned char * index_transfer(unsigned char *whole_index_table,int *index_num);
void g_block_adjust(int *B);
void change_ratio_transfer(int *index_num);
void debug_byte(char c);
void helper_array_calc();
void free_all();
void read_hdf5_input_file();
void read_hdf5_float_input_file();
void read_hdf5_double_input_file();
void read_pnetcdf_input_file();
void cmd_helper();
void assign_index(double *E);
void index_table_alignment(int *B);
void bits_packing(int *B);  /* bits operation */
void compress_index(void);


double timing[32];

int main(int argc,char *argv[])
{
	extern char * optarg;
	char *method;
	int c;
    //default B value
    int *B;
    B = (int *)malloc(1*sizeof(int));
    *B = 8;

    //default error rate
    double *E;
    E = (double *)malloc(1*sizeof(double));
    *E = 0.0010;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Barrier(MPI_COMM_WORLD);

	WRITE_FLAG = 1;
	DEBUG = 0;
	float_flag = 1;
	input_format = 0;
	output_format = 0;
	method = NULL;
	input_file_1 = NULL;
	input_file_2 = NULL;
	output_file = NULL;

	while((c = getopt(argc,argv,"i:j:o:b:e:v:m:s:hdxy"))!=-1)
	{
		switch(c)
		{
			case 'i':
				input_file_1 = strdup(optarg);
				break;
			case 'j':
				input_file_2 = strdup(optarg);
				break;
			case 'o':
				output_file = strdup(optarg);
				break;
			case 'b':
				*B = atoi(optarg);
				break;
			case 'e':
				*E = atof(optarg);
				break;
			case 'v':
				var_name = strdup(optarg);
				break;
			case 'd':
				float_flag = 0;
				break;
			case 'x':
				input_format = 1;
				break;
			case 'y':
				output_format = 1;
				break;
			case 'h':
				cmd_helper();
				break;
			case 'm':
				method = strdup(optarg);
				break;
			case 's':
				block_length = atoi(optarg);
				break;
		}
	}

	if(input_file_1==NULL)
	{
		printf("input file 1 empty!\n");
		exit(1);
	}

	if(input_file_2==NULL)
	{
		printf("input file 2 empty!\n");
		exit(1);
	}

	if(output_file==NULL)
	{
		printf("output file empty!\n");
		exit(1);
	}

	if(var_name==NULL)
	{
		printf("var name empty!\n");
		exit(1);
	}

	if(method!=NULL)
	{
		if(!strcmp(method,"equal"))
		{
			compression_method = 1;
		}
		else if(!strcmp(method,"log"))
		{
			compression_method = 3;
		}
		else if(!strcmp(method,"kmeans"))
		{
			compression_method = 0;
		}
		else if(!strcmp(method,"topk"))
		{
			compression_method = 2;
		}
		else
		{
			printf("error method!\n");
			exit(1);
		}
	}

	char path[128];
	char ch[5];
	strcpy(path,output_file);

	remove(path);

	init_bin_time = 0;
	top_k_selection_time = 0;
	cluster_table_can_reuse = 0;
	local_bit_runtime = 0;
	local_zlib_runtime = 0;

	class_num = pow(2,*B);


	MPI_Barrier(MPI_COMM_WORLD);
	timing[0] = MPI_Wtime();
#ifdef _ADD_PNETCDF_
	if(input_format==0)
		read_pnetcdf_input_file();
#endif
#ifdef _ADD_HDF5_
	if(input_format==1)
		read_hdf5_input_file();
#endif
	timing[0] = MPI_Wtime() - timing[0];


	/* phase 1: calculate change ratios */
	MPI_Barrier(MPI_COMM_WORLD);
	timing[1] = MPI_Wtime();
	calc_change_ratio(E);
	timing[1] = MPI_Wtime() - timing[1];

	/* phase 2: binning */
	MPI_Barrier(MPI_COMM_WORLD);
	timing[2] = MPI_Wtime();
	binning(B,E);
	timing[2] = MPI_Wtime() - timing[2];


	/* phase 3: indexing: assign index to each data point */
	MPI_Barrier(MPI_COMM_WORLD);
	timing[3] = MPI_Wtime();
	assign_index(E);
	timing[3] = MPI_Wtime() - timing[3];

//	free(change_ratio_flag_array);
	free(t_change_ratio_array);
	free(change_ratio_array);
//	free(data_array1);
	free(data_array2);

	/* phase 3: indexing: exchange index with neighbors */
	MPI_Barrier(MPI_COMM_WORLD);
	timing[4] = MPI_Wtime();
	index_table_alignment(B);
	timing[4] = MPI_Wtime() - timing[4];

	/* phase 3: indexing: bit-write index */
	MPI_Barrier(MPI_COMM_WORLD);
	timing[5] = MPI_Wtime();
	bits_packing(B);
	timing[5] = MPI_Wtime() - timing[5];
	free(membership);

	/* phase 3: indexing: compress index table*/
	MPI_Barrier(MPI_COMM_WORLD);
	timing[6] = MPI_Wtime();
	compress_index();
	timing[6] = MPI_Wtime() - timing[6];

	/* phase 4: write to file */
	MPI_Barrier(MPI_COMM_WORLD);
	timing[7] = MPI_Wtime();
	if(WRITE_FLAG)
	{
#ifdef _ADD_PNETCDF_
	if(output_format==0)
		pnetcdf_write();
#endif
#ifdef _ADD_HDF5_
	if(output_format==1)
		hdf5_write();
#endif
	}
	timing[7] = MPI_Wtime() - timing[7];

	double max_timing[32];

	timing[8] = data_array_row_num;
	timing[9] = -1.0*data_array_row_num;

	timing[10] = block_array_size;
	timing[11] = -1.0*block_array_size;

	int g_block_array_size;
	MPI_Allreduce(timing, max_timing, 32, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&block_array_size,&g_block_array_size,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

	if(rank==0)
	{
//		printf("ave_ratio_error: %Lf\n",ave_ratio_error); this is for DEBUG
		printf("zlib compression ratio: %Lf\n",zlib_compression_ratio);
		printf("read file time: %f\n",max_timing[0]);
		printf("calc change ratio time: %f\n",max_timing[1]);
		printf("bining time: %f\n",max_timing[2]);
		printf("assig index time: %f\n",max_timing[3]);
		printf("trans index time: %f\n",max_timing[4]);
		printf("bits opt time: %f\n",max_timing[5]);
		printf("zlib time: %f\n",max_timing[6]);
		printf("write file time: %f\n",max_timing[7]);

		printf("compression ratio: %f\n",(double)total_raw_file_size/(double)last_file_size);
//		printf("%ld\n",init_bin_time);
//		printf("%ld\n",top_k_selection_time);

//		printf("element max = %f   min = %f\n",max_timing[8],-1.0*max_timing[9]);
//		printf("block max = %f   min = %f\n",max_timing[10],-1.0*max_timing[11]);

//		printf("calc max & min = %f\n",max_timing[16]);
//		printf("construct local hist = %f\n",max_timing[17]);
//		printf("global hist reduction = %f\n",max_timing[18]);
//		printf("top-k selection = %f\n",max_timing[19]);
//		printf("prediction = %f\n",max_timing[20]);
//		printf("new cluster set up = %f\n",max_timing[21]);
//		printf("update memship = %f\n",max_timing[22]);

	}

	free_all();

	MPI_Finalize();

	return 0;
}

void compress_index(void)
{
	int i;

	// call zlib to compress each of index table block

	if(index_table_com_flag==1)
	{
		for(i=0;i<block_array_size;i++)
			compress_block(i);
	}
}

void index_table_alignment(int *B)
{
	int t_g_index_num;
	int index_num;
	int i;

	/* adjust number of elements according to boarder */

	g_block_adjust(B);
	data_split();

	double t_change_ratio_transfer;
	if (DEBUG) t_change_ratio_transfer = MPI_Wtime();

	/* MPI index transfer */

	change_ratio_transfer(&index_num);

	if (DEBUG) t_change_ratio_transfer = MPI_Wtime() - t_change_ratio_transfer;

	if (DEBUG) MPI_Allreduce(&index_num,&t_g_index_num,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

	// init meta data for index table blocks. meta data includes number of data points in the block & buffer pointers

	block_array_size = index_num/block_size;
	if(index_num%block_size)
	{
		block_array_size++;
	}

	block_array = (struct block_element *)malloc(block_array_size*sizeof(struct block_element));

	for(i=0;i<block_array_size;i++)
	{
		block_array[i].start_index = i*block_size;
		if(((i+1)*block_size)<index_num)
			block_array[i].data_num = block_size;
		else
			block_array[i].data_num = index_num - (i*block_size);
		block_array[i].uncom_buff_length = 0;
		block_array[i].com_buff_length = 0;
		block_array[i].uncom_buff = NULL;
		block_array[i].com_buff = NULL;
	}

}

void assign_index(double *E)
{
	long t_length;
	int uncompress_table_index;
	int uncompress_data_point_per_block;
	int change_ratio_index;
	int limit_count;
	limit_count = 1024;
	double ave_error = 0.0;
	int compressible_data_num_for_binning = 0;
	int g_compressible_data_num_for_binning = 0;

	int i;

	/* update index according to binning output */

	update_cluster_centroids_and_membership(E);
	free(t_membership);

	uncompress_table = (double *)malloc(limit_count*sizeof(double));
	uncompress_count = 0;

	/* find the incompressible data elements */

	for(i=0;i<data_array_row_num;i++)
	{
		if(DEBUG && diff_double(change_ratio_array[i],cluster_centroids[membership[i]])<=*E)
			ave_error += diff_double(change_ratio_array[i],cluster_centroids[membership[i]]);
		
		if(change_ratio_array[i]>=max_change_ratio || change_ratio_array[i]<=min_change_ratio)
			membership[i] = class_num-1;

		if(membership[i]!=0)
		{
			if(diff_double(change_ratio_array[i],cluster_centroids[membership[i]])>*E)
			{
				if(uncompress_count>=limit_count)
				{
					uncompress_table = (double *)realloc(uncompress_table,(limit_count+1024)*sizeof(double));
					limit_count += 1024;
				}
				uncompress_table[uncompress_count++] = data_array2[i];
				membership[i] = class_num - 1;
			}
		}

		if(membership[i]>0&&membership[i]<class_num-1)
		{
			compressible_data_num_for_binning ++;
		}
	}

//	if(rank==0)
//		printf("%d\n",g_compressible_data_num_for_binning);
//	return ;

	if (DEBUG) {
		long double g_error;
		long long g_data_array_row_num;
		long long t_num;
		long double t_error;

		t_error = ave_error;

		MPI_Allreduce(&t_error,&g_error,1,MPI_LONG_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

		t_num = data_array_row_num;

		MPI_Allreduce(&t_num,&g_data_array_row_num,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);

		ave_ratio_error = g_error/g_data_array_row_num;

//		if(rank==0)
//			printf("ave_ratio_error: %Lf\n",g_error/g_data_array_row_num);

		MPI_Barrier(MPI_COMM_WORLD);
	}
}

//helper cmd
void cmd_helper()
{
	char *help =
		"Usage: mpirun -n 'number_of_process' ./exe_mpi_compress -i 'input_file_name_1' -j 'input_file_name_2' -o 'output_file_name' -v 'varable name'\n"
		"-i input_file_name_1	: first input file name\n"
		"-j input_file_name_2	: second input file name\n"
		"-o output_file_name	: output file name\n"
		"-v varable name		: varable name in input file\n"
		"-m binning_method	: equal (equal bining)/ log (log scaled binning)/ kmeans (kmeans binning)/ topk (top-k binning, default)\n"
		"-e error_rate		: user-defined threshold (default 0.1%)\n"
		"-d			: elements are double data type (default is float)\n"
		"-x			: input file format is hdf5 (default is Pnetcdf)\n"
		"-y			: output file format is hdf5 (default is Pnetcdf)\n"
		"-s block_size		: set index block size(Byte) default:1024"
		"-b B			: set index lenth B (not for top-k)"
		"-h			: print this help information\n";

	if(rank==0)
		printf("%s\n",help);
}

//free memory space
void free_all()
{
	int i;

	free(cluster_centroids);

	for(i=0;i<block_array_size;i++)
	{
//		free(block_array[i].uncom_buff);
//		free(block_array[i].com_buff);
	}

	free(uncompress_table);
	free(block_array);
//	free(membership);
//	free(t_membership);
//	free(change_ratio_array);
//	free(change_ratio_flag_array);
//	free(t_change_ratio_array);
	free(g_block_map);
//	free(data_array1);
//	free(data_array2);


// free B
}

#ifdef _ADD_HDF5_

//write compressed NUMARCK data into hdf5 file
void hdf5_write()
{
	struct timeval t_start,t_end;
	long g_min_local_runtime,g_max_local_runtime;
	long long g_data_array_row_num;
	int file_flag = 1;
	long cluster_centroids_table_size,block_table_size,index_table_size,uncompress_table_size;
	long g_cluster_centroids_table_size,g_block_table_size,g_index_table_size,g_uncompress_table_size;
	int varid_g_data_array_row_num,varid_file_flag,varid_class_num,varid_index_table_com_flag,varid_block_size,varid_g_block_num;
	int varid_cluster_table,varid_block_table_length,varid_block_table_uncompress_count,varid_index_table,varid_uncompress_table,varid_info;
	int ret,ncfile,dimid0, ndims=1,dimid2;
	int dimid_cluster_table,dimid_block_table,dimid_index_table,dimid_uncompress_table;
	int l_uncompress_sum,t_uncompress_sum;
	MPI_Offset start, count=1;
	MPI_Offset index_0[1];
	int flag = 0;
	double d[2] = {1.1,2.2};
	double data;
	int i;

	/*
	gettimeofday(&index_table_compression_start,NULL);
	gettimeofday(&t_start,NULL);
	if(index_table_com_flag==1)
	{
		for(i=0;i<block_array_size;i++)
			compress_block(i);

	}
	gettimeofday(&phase_3_end,NULL);
	gettimeofday(&t_end,NULL);
	local_zlib_runtime += time_diff(t_start,t_end);

	long local_runtime;
	long g_min_local_bit_runtime;
	long g_max_local_bit_runtime;
	long g_min_local_zlib_runtime;
	long g_max_local_zlib_runtime;
	local_runtime = local_bit_runtime + local_zlib_runtime;

	int g_min_size,g_max_size;

	MPI_Allreduce(&block_array_size,&g_min_size,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
	MPI_Allreduce(&block_array_size,&g_max_size,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
	MPI_Allreduce(&local_runtime,&g_min_local_runtime,1,MPI_LONG,MPI_MIN,MPI_COMM_WORLD);
	MPI_Allreduce(&local_runtime,&g_max_local_runtime,1,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);
	MPI_Allreduce(&local_bit_runtime,&g_min_local_bit_runtime,1,MPI_LONG,MPI_MIN,MPI_COMM_WORLD);
	MPI_Allreduce(&local_bit_runtime,&g_max_local_bit_runtime,1,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);
	MPI_Allreduce(&local_zlib_runtime,&g_min_local_zlib_runtime,1,MPI_LONG,MPI_MIN,MPI_COMM_WORLD);
	MPI_Allreduce(&local_zlib_runtime,&g_max_local_zlib_runtime,1,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&write_file_start,NULL);
	*/


//return ;

//	gettimeofday(&write_file_end,NULL);
	
	// calc global number of data point 
	
	long long t_num;
	t_num = data_array_row_num;
	MPI_Allreduce(&t_num,&g_data_array_row_num,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);

	// calc global size of different tables

	cluster_centroids_table_size = class_num*sizeof(double);
	g_cluster_centroids_table_size = class_num*sizeof(double);

	MPI_Allreduce(&block_array_size,&g_block_num,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);


	index_table_size = 0;
	long uncompress_index_table_size = 0;
	long g_uncompress_index_table_size = 0;

	for(i=0;i<block_array_size;i++)
	{
		if(index_table_com_flag==1)
			index_table_size += block_array[i].com_buff_length;
		else
			index_table_size += block_array[i].uncom_buff_length;

		uncompress_index_table_size += block_array[i].uncom_buff_length;
	}

//	printf("zlib compression ratio   = %Lf\n",(long double)(g_data_array_row_num)*(long double)(B/8)/(long double)index_table_size);

//	printf("total num = %d,%d,%ld\n",g_data_array_row_num,B,index_table_size);
	MPI_Allreduce(&index_table_size,&g_index_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&uncompress_index_table_size,&g_uncompress_index_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);

	zlib_compression_ratio = (long double)g_uncompress_index_table_size/(long double)g_index_table_size;
//	if(rank==0)
//		printf("zlib compression ratio = %Lf\n",(long double)g_uncompress_index_table_size/(long double)g_index_table_size);

	if(float_flag==0)
	{
		uncompress_table_size = uncompress_count*sizeof(double);
		MPI_Allreduce(&uncompress_table_size,&g_uncompress_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	}
	else
	{
		uncompress_table_size = uncompress_count*sizeof(float);
		MPI_Allreduce(&uncompress_table_size,&g_uncompress_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	}

	char path[128];
	char t_ch[64];
	char ch[5];

	strcpy(path,output_file);
	
	// hdf5 start

	hid_t plist_id;
	hid_t file_id,dset_id;
	hid_t ret_id;

	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	assert(plist_id>=0);
	H5Pset_fapl_mpio(plist_id,MPI_COMM_WORLD,MPI_INFO_NULL);

	file_id = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	assert(file_id>=0);
	H5Pclose(plist_id);
	

hid_t filespace_info,filespace_bin_centers,filespace_block_table,filespace_index_table,filespace_incompressible_table;
	hsize_t hdim;

	// create different tables in NUMARCK file and define the sizes of different table

	hdim = 1;
	filespace_info = H5Screate_simple(1,&hdim,NULL);
	assert(filespace_info>=0);

	hdim = g_cluster_centroids_table_size/sizeof(double);
	filespace_bin_centers = H5Screate_simple(1,&hdim,NULL);
	assert(filespace_bin_centers>=0);

	hdim = g_block_num;
	filespace_block_table = H5Screate_simple(1,&hdim,NULL);
	assert(filespace_block_table>=0);

	hdim = g_index_table_size;
	filespace_index_table = H5Screate_simple(1,&hdim,NULL);
	assert(filespace_index_table>=0);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_incompressible_table_dim");
	int zero_uncompress_flag;
	if(g_uncompress_table_size)
	{
		if(float_flag==0)
		{
			hdim = g_uncompress_table_size/sizeof(double);
			filespace_incompressible_table = H5Screate_simple(1,&hdim,NULL);

			incompressible_ratio = g_uncompress_table_size/sizeof(double)*100/(long double)g_data_array_row_num;
			
//			if(rank==0)
//				printf("incompressible ratio %Lf\n",g_uncompress_table_size/sizeof(double)*100/(long double)g_data_array_row_num);
		}
		else
		{
			hdim = g_uncompress_table_size/sizeof(float);
			filespace_incompressible_table = H5Screate_simple(1,&hdim,NULL);

			incompressible_ratio = g_uncompress_table_size/sizeof(float)*100/(long double)g_data_array_row_num;

//			if(rank==0)
//				printf("incompressible ratio %Lf\n",g_uncompress_table_size/sizeof(float)*100/(long double)g_data_array_row_num);
		}
			zero_uncompress_flag = 0;
	}
	else
	{
		hdim = 1;
		filespace_incompressible_table = H5Screate_simple(1,&hdim,NULL);

		zero_uncompress_flag = 1;
	}

//	MPI_Barrier(MPI_COMM_WORLD);

	count = class_num;
	start = BLOCK_LOW(rank,size,count);
	count = BLOCK_SIZE(rank,size,count);

	hsize_t h_count,h_offset;
	hid_t memspace;
	herr_t status;

	// note that the raw data type can be float or double the data type of bin center & incompressible data table are also float or double depending on the raw data type.
	// each MPI process then collective write different tables.

	if(float_flag==0)
	{
		double *t_centers = (double *)malloc(class_num*sizeof(double));
		for(i=0;i<class_num;i++)
			t_centers[i] = cluster_centroids[i];

		dset_id = H5Dcreate(file_id,"cluster_table",H5T_NATIVE_DOUBLE,filespace_bin_centers,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
		assert(dset_id>=0);

		h_offset = start;
		h_count = count;

		memspace = H5Screate_simple(1,&h_count,NULL);
		assert(memspace>=0);

		H5Sselect_hyperslab(filespace_bin_centers,H5S_SELECT_SET,&h_offset,NULL,&h_count,NULL);

		plist_id = H5Pcreate(H5P_DATASET_XFER);
		assert(plist_id>=0);
		H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		status = H5Dwrite(dset_id,H5T_NATIVE_DOUBLE,memspace,filespace_bin_centers,plist_id,t_centers+start);
		assert(status>=0);
		H5Sclose(memspace);
		H5Dclose(dset_id);
		H5Sclose(filespace_bin_centers);

		free(t_centers);

	}
	else
	{
		float *t_centers = (float *)malloc(class_num*sizeof(float));
		for(i=0;i<class_num;i++)
			t_centers[i] = cluster_centroids[i];

		dset_id = H5Dcreate(file_id,"cluster_table",H5T_NATIVE_FLOAT,filespace_bin_centers,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
		assert(dset_id>=0);

		h_offset = start;
		h_count = count;

		memspace = H5Screate_simple(1,&h_count,NULL);
		assert(memspace>=0);

		ret_id = H5Sselect_hyperslab(filespace_bin_centers,H5S_SELECT_SET,&h_offset,NULL,&h_count,NULL);
		assert(ret_id>=0);

		plist_id = H5Pcreate(H5P_DATASET_XFER);
		assert(plist_id);
		ret_id = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
		assert(ret_id>=0);

		status = H5Dwrite(dset_id,H5T_NATIVE_FLOAT,memspace,filespace_bin_centers,plist_id,t_centers+start);
		assert(status>=0);
		H5Sclose(memspace);
		H5Dclose(dset_id);
		H5Sclose(filespace_bin_centers);

		free(t_centers);
	}

//	MPI_Barrier(MPI_COMM_WORLD);

	long t_offset;
	int *length_array;
	int *uncompress_count_array;
	long ttt_offset;
	long t_block_array_size;

	t_block_array_size = block_array_size;

	MPI_Scan(&block_array_size,&ttt_offset,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	ttt_offset -=block_array_size;

	start = ttt_offset;
	count = block_array_size;

	length_array = (int *)malloc(block_array_size*sizeof(int));
	for(i=0;i<block_array_size;i++)
	{
		if(index_table_com_flag==1)
			length_array[i] = block_array[i].com_buff_length;
		else
			length_array[i] = block_array[i].uncom_buff_length;
	}

	dset_id = H5Dcreate(file_id,"block_table_length",H5T_NATIVE_INT,filespace_block_table,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

	h_offset = start;
	h_count = count;

	memspace = H5Screate_simple(1,&h_count,NULL);
	assert(memspace>=0);

	H5Sselect_hyperslab(filespace_block_table,H5S_SELECT_SET,&h_offset,NULL,&h_count,NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	assert(plist_id>=0);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	status = H5Dwrite(dset_id,H5T_NATIVE_INT,memspace,filespace_block_table,plist_id,length_array);
	assert(status>=0);
//	H5Sclose(filespace_block_table);
	H5Sclose(memspace);
	H5Dclose(dset_id);

	free(length_array);


	l_uncompress_sum = 0;
	for(i=0;i<block_array_size;i++)
	{
		l_uncompress_sum +=block_array[i].uncompress_data_point_num;
	}

	MPI_Scan(&l_uncompress_sum,&t_uncompress_sum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	t_uncompress_sum -=l_uncompress_sum;


//	MPI_Barrier(MPI_COMM_WORLD);

	for(i=0;i<block_array_size;i++)
	{
		if(i==0)
			block_array[i].previous_data_point_num = t_uncompress_sum;
		else
			block_array[i].previous_data_point_num = block_array[i-1].previous_data_point_num+block_array[i-1].uncompress_data_point_num;
	}

	uncompress_count_array = (int *)malloc(block_array_size*sizeof(int));
	for(i=0;i<block_array_size;i++)
		uncompress_count_array[i] = block_array[i].previous_data_point_num;

	dset_id = H5Dcreate(file_id,"block_table_uncompress_count",H5T_NATIVE_INT,filespace_block_table,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
	assert(dset_id>=0);

	h_offset = start;
	h_count = count;

	memspace = H5Screate_simple(1,&h_count,NULL);
	assert(memspace>=0);

	H5Sselect_hyperslab(filespace_block_table,H5S_SELECT_SET,&h_offset,NULL,&h_count,NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	assert(plist_id>=0);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	status = H5Dwrite(dset_id,H5T_NATIVE_INT,memspace,filespace_block_table,plist_id,uncompress_count_array);
	assert(status>=0);
	H5Sclose(filespace_block_table);
	H5Sclose(memspace);
	H5Dclose(dset_id);


	free(uncompress_count_array);

	char *index_table_array;
	long tt_offset;

	MPI_Scan(&index_table_size,&t_offset,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	t_offset -=index_table_size;

	start = t_offset;
	count = index_table_size;

	index_table_array = (char *)malloc(index_table_size*sizeof(char));

	tt_offset = 0;

	for(i=0;i<block_array_size;i++)
	{
		if(index_table_com_flag==1)
		{
			memcpy(&index_table_array[tt_offset],block_array[i].com_buff,block_array[i].com_buff_length);
			tt_offset += block_array[i].com_buff_length;
		}
		else
		{
			memcpy(&index_table_array[tt_offset],block_array[i].uncom_buff,block_array[i].uncom_buff_length);
			tt_offset += block_array[i].uncom_buff_length;
		}
	}

	dset_id = H5Dcreate(file_id,"block_table",H5T_NATIVE_CHAR,filespace_index_table,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

	h_offset = start;
	h_count = count;

	memspace = H5Screate_simple(1,&h_count,NULL);

	H5Sselect_hyperslab(filespace_index_table,H5S_SELECT_SET,&h_offset,NULL,&h_count,NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	status = H5Dwrite(dset_id,H5T_NATIVE_CHAR,memspace,filespace_index_table,plist_id,index_table_array);
	assert(status>=0);
	H5Sclose(filespace_index_table);
	H5Sclose(memspace);
	H5Dclose(dset_id);

	free(index_table_array);

	MPI_Scan(&uncompress_table_size,&t_offset,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	t_offset -=uncompress_table_size;


	if(float_flag==0)
	{
		start = t_offset/sizeof(double);
		if(zero_uncompress_flag==0)
			count = uncompress_table_size/sizeof(double);
		else
			count = 1;
	}
	else
	{
		start = t_offset/sizeof(float);
		if(zero_uncompress_flag==0)
			count = uncompress_table_size/sizeof(float);
		else
			count = 1;
	}

	if(float_flag==1)
	{
		if(zero_uncompress_flag==0)
		{
			float *float_uncompress_table;

			float_uncompress_table = (float *)malloc(uncompress_count*sizeof(float));
			for(i=0;i<uncompress_count;i++)
				float_uncompress_table[i] = uncompress_table[i];

			dset_id = H5Dcreate(file_id,"incompressible_table",H5T_NATIVE_FLOAT,filespace_incompressible_table,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
			assert(dset_id>=0);

			h_offset = start;
			h_count = count;

			memspace = H5Screate_simple(1,&h_count,NULL);
			assert(memspace>=0);

			ret_id = H5Sselect_hyperslab(filespace_incompressible_table,H5S_SELECT_SET,&h_offset,NULL,&h_count,NULL);
			assert(ret_id>=0);

			plist_id = H5Pcreate(H5P_DATASET_XFER);
			assert(plist_id>=0);
			ret_id = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			assert(ret_id>=0);

			status = H5Dwrite(dset_id,H5T_NATIVE_FLOAT,memspace,filespace_incompressible_table,plist_id,float_uncompress_table);
			assert(status>=0);
			H5Sclose(filespace_incompressible_table);
			H5Sclose(memspace);
			H5Dclose(dset_id);

			free(float_uncompress_table);
		}
		else
		{
			//caution! if part changed
			float t_float = 1234.56789;

			dset_id = H5Dcreate(file_id,"incompressible_table",H5T_NATIVE_FLOAT,filespace_incompressible_table,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
			plist_id = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

			status = H5Dwrite(dset_id,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,plist_id,&t_float);
			assert(status>=0);
			H5Dclose(dset_id);
		}
	}
	else
	{
		if(zero_uncompress_flag==0)
		{
			dset_id = H5Dcreate(file_id,"incompressible_table",H5T_NATIVE_DOUBLE,filespace_incompressible_table,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

			h_offset = start;
			h_count = count;

			memspace = H5Screate_simple(1,&h_count,NULL);

			H5Sselect_hyperslab(filespace_incompressible_table,H5S_SELECT_SET,&h_offset,NULL,&h_count,NULL);

			plist_id = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

			status = H5Dwrite(dset_id,H5T_NATIVE_DOUBLE,memspace,filespace_incompressible_table,plist_id,uncompress_table);
			assert(status>=0);
			H5Sclose(filespace_incompressible_table);
			H5Sclose(memspace);
			H5Dclose(dset_id);

		}
		else
		{
			double t_double = 1234.5678910;

			dset_id = H5Dcreate(file_id,"incompressible_table",H5T_NATIVE_DOUBLE,filespace_incompressible_table,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
			plist_id = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

			status = H5Dwrite(dset_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,plist_id,&t_double);
			assert(status>=0);
			H5Dclose(dset_id);
		}
	}

	H5Fclose(file_id);

//	MPI_Barrier(MPI_COMM_WORLD);

	if(rank==0)
	{
		struct stat st; 
    		if(stat(path, &st) == 0)
		{
			if(float_flag==1)
			{
//				printf("%lf\t",g_data_array_row_num*sizeof(float)/(double)st.st_size);
				total_raw_file_size += g_data_array_row_num*sizeof(float);
			}
			else
			{
//				printf("%lf\t",g_data_array_row_num*sizeof(double)/(double)st.st_size);
				total_raw_file_size += g_data_array_row_num*sizeof(double);
			}
			last_file_size = st.st_size;
		}
		else
		{
			printf("get file size error\n");
		}
	}
}
#endif

#ifdef _ADD_PNETCDF_

//write compressed NUMARCK data into pnetcdf file
void pnetcdf_write()
{
	struct timeval t_start,t_end;
	long g_min_local_runtime,g_max_local_runtime;
	int g_data_array_row_num;
	int file_flag = 1;
	long cluster_centroids_table_size,block_table_size,index_table_size,uncompress_table_size;
	long g_cluster_centroids_table_size,g_block_table_size,g_index_table_size,g_uncompress_table_size;
	int varid_g_data_array_row_num,varid_file_flag,varid_class_num,varid_index_table_com_flag,varid_block_size,varid_g_block_num;
	int varid_cluster_table,varid_block_table_length,varid_block_table_uncompress_count,varid_index_table,varid_uncompress_table,varid_info;
	int ret,ncfile,dimid0, ndims=1,dimid2;
	int dimid_cluster_table,dimid_block_table,dimid_index_table,dimid_uncompress_table;
	int l_uncompress_sum,t_uncompress_sum;
	MPI_Offset start, count=1;
	MPI_Offset index_0[1];
	int flag = 0;
	double d[2] = {1.1,2.2};
	double data;
	int i;

//	MPI_Barrier(MPI_COMM_WORLD);
//return ;

	MPI_Allreduce(&data_array_row_num,&g_data_array_row_num,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

	cluster_centroids_table_size = class_num*sizeof(double);
	g_cluster_centroids_table_size = class_num*sizeof(double);

//	block_table_size = block_array_size*(2*sizeof(long)+sizeof(int));
//	MPI_Allreduce(&block_table_size,&g_block_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&block_array_size,&g_block_num,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);


	index_table_size = 0;
	long uncompress_index_table_size = 0;
	long g_uncompress_index_table_size = 0;

	for(i=0;i<block_array_size;i++)
	{
		if(index_table_com_flag==1)
			index_table_size += block_array[i].com_buff_length;
		else
			index_table_size += block_array[i].uncom_buff_length;

		uncompress_index_table_size += block_array[i].uncom_buff_length;
	}

//	printf("zlib compression ratio   = %Lf\n",(long double)(g_data_array_row_num)*(long double)(B/8)/(long double)index_table_size);

//	printf("total num = %d,%d,%ld\n",g_data_array_row_num,B,index_table_size);
	MPI_Allreduce(&index_table_size,&g_index_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&uncompress_index_table_size,&g_uncompress_index_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);

	zlib_compression_ratio = (long double)g_uncompress_index_table_size/(long double)g_index_table_size;
//	if(rank==0)
//		printf("zlib compression ratio = %Lf\n",(long double)g_uncompress_index_table_size/(long double)g_index_table_size);

	if(float_flag==0)
	{
		uncompress_table_size = uncompress_count*sizeof(double);
		MPI_Allreduce(&uncompress_table_size,&g_uncompress_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	}
	else
	{
		uncompress_table_size = uncompress_count*sizeof(float);
		MPI_Allreduce(&uncompress_table_size,&g_uncompress_table_size,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	}

	char path[128];
	char t_ch[64];
	char ch[5];

	strcpy(path,output_file);

	ret = ncmpi_create(MPI_COMM_WORLD, path, NC_NOCLOBBER|NC_64BIT_OFFSET, MPI_INFO_NULL, &ncfile);
//	ret = ncmpi_create(MPI_COMM_WORLD, path, NC_CLOBBER|NC_64BIT_OFFSET, MPI_INFO_NULL, &ncfile);
	if (ret != NC_NOERR && ret !=NC_EEXIST) handle_error(ret, __LINE__);
	if(ret==NC_EEXIST)
	{
		ret = ncmpi_open(MPI_COMM_WORLD, path, NC_WRITE,MPI_INFO_NULL,&ncfile);
		if (ret != NC_NOERR) handle_error(ret, __LINE__);
		ret = ncmpi_redef(ncfile);
	}
	
	strcpy(t_ch,var_name);
	strcat(t_ch,"_info_dim");
	ret = ncmpi_def_dim(ncfile, t_ch, 1, &dimid0);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_bin_centers_dim");
	ret = ncmpi_def_dim(ncfile, t_ch, g_cluster_centroids_table_size/sizeof(double), &dimid_cluster_table);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_block_table_dim");
	ret = ncmpi_def_dim(ncfile, t_ch, g_block_num, &dimid_block_table);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_index_table_dim");
	ret = ncmpi_def_dim(ncfile, t_ch, g_index_table_size, &dimid_index_table);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_incompressible_table_dim");
	int zero_uncompress_flag;
	if(g_uncompress_table_size)
	{
		if(float_flag==0)
		{
			ret = ncmpi_def_dim(ncfile, t_ch, g_uncompress_table_size/sizeof(double), &dimid_uncompress_table);
			if(ret != NC_NOERR) handle_error(ret, __LINE__);

			incompressible_ratio = g_uncompress_table_size/sizeof(double)*100/(long double)g_data_array_row_num;
//			if(rank==0)
//				printf("incompressible ratio %Lf\n",g_uncompress_table_size/sizeof(double)*100/(long double)g_data_array_row_num);
		}
		else
		{
			ret = ncmpi_def_dim(ncfile, t_ch, g_uncompress_table_size/sizeof(float), &dimid_uncompress_table);
			if(ret != NC_NOERR) handle_error(ret, __LINE__);

			incompressible_ratio = g_uncompress_table_size/sizeof(float)*100/(long double)g_data_array_row_num;
//			if(rank==0)
//				printf("incompressible ratio %Lf\n",g_uncompress_table_size/sizeof(float)*100/(long double)g_data_array_row_num);
		}
			zero_uncompress_flag = 0;
	}
	else
	{
		ret = ncmpi_def_dim(ncfile, t_ch, 1, &dimid_uncompress_table);
		if(ret != NC_NOERR) handle_error(ret, __LINE__);
		zero_uncompress_flag = 1;
	}

	int t_int;
	t_int = 0;
	strcpy(t_ch,var_name);
	strcat(t_ch,"_info");
	ret = ncmpi_def_var(ncfile, t_ch, NC_INT, ndims, &dimid0, &varid_info);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_put_att_int(ncfile,varid_info,"total_data_num",NC_INT,1,&g_data_array_row_num);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_put_att_int(ncfile,varid_info,"is_var_compress",NC_INT,1,&file_flag);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_put_att_int(ncfile,varid_info,"centers_number",NC_INT,1,&class_num);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_put_att_int(ncfile,varid_info,"is_index_table_compress",NC_INT,1,&index_table_com_flag);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_put_att_int(ncfile,varid_info,"elements_per_block",NC_INT,1,&block_size);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_cluster_table");
	if(float_flag==0)
	{
		ret = ncmpi_def_var(ncfile, t_ch, NC_DOUBLE, ndims, &dimid_cluster_table, &varid_cluster_table);
		if(ret != NC_NOERR) handle_error(ret, __LINE__);
	}
	else
	{
		ret = ncmpi_def_var(ncfile, t_ch, NC_FLOAT, ndims, &dimid_cluster_table, &varid_cluster_table);
		if(ret != NC_NOERR) handle_error(ret, __LINE__);
	}

//	char special[] = "123456789 in index table correspond to the uncompressiable data index (usally, 2^B-1).";
//	ret = ncmpi_put_att_text (ncfile, NC_GLOBAL, "Speciall constant for index table", strlen(special), special);
//	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_index_table_offset");
	ret = ncmpi_def_var(ncfile, t_ch, NC_INT, ndims, &dimid_block_table, &varid_block_table_length);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_incompressible_table_offset");
	ret = ncmpi_def_var(ncfile, t_ch, NC_INT, ndims, &dimid_block_table, &varid_block_table_uncompress_count);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_index_table");
	ret = ncmpi_def_var(ncfile, t_ch, NC_BYTE, ndims, &dimid_index_table, &varid_index_table);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	strcpy(t_ch,var_name);
	strcat(t_ch,"_incompressible_offset");
	if(float_flag==0)
	{
		ret = ncmpi_def_var(ncfile, t_ch, NC_DOUBLE, ndims, &dimid_uncompress_table, &varid_uncompress_table);
		if(ret != NC_NOERR) handle_error(ret, __LINE__);
	}
	else
	{
		ret = ncmpi_def_var(ncfile, t_ch, NC_FLOAT, ndims, &dimid_uncompress_table, &varid_uncompress_table);
		if(ret != NC_NOERR) handle_error(ret, __LINE__);
	}

	ret = ncmpi_enddef(ncfile);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

//	MPI_Barrier(MPI_COMM_WORLD);


	count = class_num;
	start = BLOCK_LOW(rank,size,count);
	count = BLOCK_SIZE(rank,size,count);

	if(float_flag==0)
	{
		double *t_centers = (double *)malloc(class_num*sizeof(double));
		for(i=0;i<class_num;i++)
			t_centers[i] = cluster_centroids[i];

		ret = ncmpi_put_vara_double_all(ncfile, varid_cluster_table, &start, &count, t_centers+start);
		if (ret != NC_NOERR) handle_error(ret, __LINE__);
		free(t_centers);
	}
	else
	{
		float *t_centers = (float *)malloc(class_num*sizeof(float));
		for(i=0;i<class_num;i++)
			t_centers[i] = cluster_centroids[i];

		ret = ncmpi_put_vara_float_all(ncfile, varid_cluster_table, &start, &count, t_centers+start);
		if (ret != NC_NOERR) handle_error(ret, __LINE__);

		free(t_centers);
	}

//	MPI_Barrier(MPI_COMM_WORLD);

	long t_offset;
	int *length_array;
	int *uncompress_count_array;
	int ttt_offset; //int

	MPI_Scan(&block_array_size,&ttt_offset,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	ttt_offset -=block_array_size;

	start = ttt_offset;
	count = block_array_size;

	length_array = (int *)malloc(block_array_size*sizeof(int));
	for(i=0;i<block_array_size;i++)
	{
		if(index_table_com_flag==1)
			length_array[i] = block_array[i].com_buff_length;
		else
			length_array[i] = block_array[i].uncom_buff_length;
	}


	ret = ncmpi_put_vara_int_all(ncfile, varid_block_table_length, &start, &count, length_array);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	free(length_array);

	l_uncompress_sum = 0;
	for(i=0;i<block_array_size;i++)
	{
		l_uncompress_sum +=block_array[i].uncompress_data_point_num;
	}

	MPI_Scan(&l_uncompress_sum,&t_uncompress_sum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	t_uncompress_sum -=l_uncompress_sum;


//	MPI_Barrier(MPI_COMM_WORLD);

	for(i=0;i<block_array_size;i++)
	{
		if(i==0)
			block_array[i].previous_data_point_num = t_uncompress_sum;
		else
			block_array[i].previous_data_point_num = block_array[i-1].previous_data_point_num+block_array[i-1].uncompress_data_point_num;
	}

	uncompress_count_array = (int *)malloc(block_array_size*sizeof(int));
	for(i=0;i<block_array_size;i++)
		uncompress_count_array[i] = block_array[i].previous_data_point_num;


	ret = ncmpi_put_vara_int_all(ncfile, varid_block_table_uncompress_count, &start, &count, uncompress_count_array);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	free(uncompress_count_array);

	char *index_table_array;
	long tt_offset;

	MPI_Scan(&index_table_size,&t_offset,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
	t_offset -=index_table_size;

	start = t_offset;
	count = index_table_size;

	index_table_array = (char *)malloc(index_table_size*sizeof(char));

	tt_offset = 0;

	for(i=0;i<block_array_size;i++)
	{
		if(index_table_com_flag==1)
		{
			memcpy(&index_table_array[tt_offset],block_array[i].com_buff,block_array[i].com_buff_length);

			tt_offset += block_array[i].com_buff_length;
		}
		else
		{
			memcpy(&index_table_array[tt_offset],block_array[i].uncom_buff,block_array[i].uncom_buff_length);
			tt_offset += block_array[i].uncom_buff_length;
		}
	}

	ret = ncmpi_put_vara_schar_all(ncfile, varid_index_table, &start, &count, index_table_array);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

	free(index_table_array);

	MPI_Scan(&uncompress_table_size,&t_offset,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	t_offset -=uncompress_table_size;


	if(float_flag==0)
	{
		start = t_offset/sizeof(double);
		if(zero_uncompress_flag==0)
			count = uncompress_table_size/sizeof(double);
		else
			count = 1;
	}
	else
	{
		start = t_offset/sizeof(float);
		if(zero_uncompress_flag==0)
			count = uncompress_table_size/sizeof(float);
		else
			count = 1;
	}

	if(float_flag==1)
	{
		if(zero_uncompress_flag==0)
		{
			float *float_uncompress_table;

			float_uncompress_table = (float *)malloc(uncompress_count*sizeof(float));
			for(i=0;i<uncompress_count;i++)
				float_uncompress_table[i] = uncompress_table[i];

			ret = ncmpi_put_vara_float_all(ncfile, varid_uncompress_table, &start, &count, float_uncompress_table);
			if(ret != NC_NOERR) handle_error(ret, __LINE__);
			free(float_uncompress_table);
		}
		else
		{
			float t_float = 1234.56789;
			ret = ncmpi_put_vara_float_all(ncfile, varid_uncompress_table, &start, &count, &t_float);
			if(ret != NC_NOERR) handle_error(ret, __LINE__);
		}
	}
	else
	{
		if(zero_uncompress_flag==0)
		{
			ret = ncmpi_put_vara_double_all(ncfile, varid_uncompress_table, &start, &count, uncompress_table);
			if(ret != NC_NOERR) handle_error(ret, __LINE__);
		}
		else
		{
			double t_double = 1234.5678910;
			ret = ncmpi_put_vara_double_all(ncfile, varid_uncompress_table, &start, &count, &t_double);
			if(ret != NC_NOERR) handle_error(ret, __LINE__);
		}
	}

	ret = ncmpi_close(ncfile);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);

//	MPI_Barrier(MPI_COMM_WORLD);

	if(rank==0)
	{
		struct stat st; 
    		if(stat(path, &st) == 0)
		{
			if(float_flag==1)
			{
//				printf("%lf\t",g_data_array_row_num*sizeof(float)/(double)st.st_size);
				total_raw_file_size += g_data_array_row_num*sizeof(float);
			}
			else
			{
//				printf("%lf\t",g_data_array_row_num*sizeof(double)/(double)st.st_size);
				total_raw_file_size += g_data_array_row_num*sizeof(double);
			}
			last_file_size = st.st_size;
		}
		else
		{
			printf("get file size error\n");
		}
	}
}
#endif

//compress index table block using zlib
void compress_block(int block_index)
{

	unsigned char *com_buff;
	int ret, flush;
	unsigned have;
	z_stream strm;
	uLong blen;

	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	ret = deflateInit(&strm, zlib_compression_level);
	if (ret != Z_OK)
	{
		printf("zlib error!\n");
		exit(1);
	}
	
	blen = block_array[block_index].uncom_buff_length;
	com_buff = (unsigned char *)malloc(blen*sizeof(unsigned char));
	strm.avail_in = block_array[block_index].uncom_buff_length;
	strm.next_in = block_array[block_index].uncom_buff;
	strm.avail_out = blen;
	strm.next_out = com_buff;

	ret = deflate(&strm, Z_FINISH);
	assert(ret != Z_STREAM_ERROR);

	block_array[block_index].com_buff_length = blen - strm.avail_out;
	block_array[block_index].com_buff = com_buff;

	ret = deflateEnd(&strm);
	assert(ret == Z_OK);
}


/* basic steps of binning()
 * prepare helper change ratio array for binning algorithm
 * call binning algorithm
 * update the outout of binning algoriothm
 * note that data points whose change ratios<E will not be considered in binning algorithms
 */

void binning(int *B, double *E)
{
	struct timeval t_start,t_end;
	double width;
	int index;
	int i;

	// malloc space for index table
	cluster_centroids = (double *)malloc((class_num)*sizeof(double));

	index = 0;

	if (DEBUG) {
		MPI_Barrier(MPI_COMM_WORLD);
		// binning_start = MPI_Wtime();
	}

	int g_t_change_ratio_array_length;

	if(compression_method==0)   /* k-means based binning method */
	{
		// select k-means init cluster

		cluster_initial_centroids_collection(t_change_ratio_array_length,t_change_ratio_array);

		MPI_Bcast(cluster_centroids, (class_num-2)*1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		block_size = block_length*8/(*B);

		// call k-means to get the bin centers

		mpi_kmeans(t_change_ratio_array,1,t_change_ratio_array_length,class_num-2,5,t_membership,cluster_centroids,MPI_COMM_WORLD);
	}
	else if(compression_method==1) /* equal-width based binning */
	{
		mpi_equal_bin(t_change_ratio_array,t_change_ratio_array_length,class_num-2,t_membership,B,E);
	}
	else if(compression_method==2) /* top-k binning */
	{
		mpi_simple_grouping2(t_change_ratio_array,t_change_ratio_array_length,class_num-2,t_membership,zero_change_ratio_num,B,E);
	}
	else if(compression_method==3) /* log based binning */
	{
		mpi_log_bin(t_change_ratio_array,t_change_ratio_array_length,class_num-2,t_membership,B,E);
	}


	return;

}

// top-k binning
// t_change_ratio_array: change_ratios input to binning strategy
// index: length of t_change_ratio_array
// r_class_num: number of bins selected by binning strategy
// t_membership: output, bin index of each data points
// zero_change_ratio_num: number of change ratios <= E

void mpi_simple_grouping2(double *t_change_ratio_array,int index,int r_class_num,int *t_membership,int zero_change_ratio_num,int *B, double *E)
{
	FILE *fp;
	struct t_center_element *t_center;
	struct t_center_element *max_center;
	struct t_center_element t;
	struct timeval t_start,t_end;
	double min,max;
	double g_min,g_max;
	double bin_width;
	double min_dis;
	int bin_pos;
	int t_group_num;
	int t_index;
	int t_mem;
	int i,j;

	timing[16] = MPI_Wtime();

	// calc local max & min change ratios

	min = t_change_ratio_array[0];
	max = t_change_ratio_array[0];

	for(i=1;i<index;i++)
	{
		if(t_change_ratio_array[i]<min)
			min = t_change_ratio_array[i];
		if(t_change_ratio_array[i]>max)
			max = t_change_ratio_array[i];
	}

	int g_zero_change_ratio_num;

	// get the global number of change ratios <= E. this value will be used to estimate NUMARCK file length.

	MPI_Allreduce(&zero_change_ratio_num,&(g_zero_change_ratio_num),1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

	double re[2],g_re[2];

	re[0] = -1.0*min;
	re[1] = max;

	// get the global range of change ratios

	MPI_Allreduce(re,g_re,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

	g_min = -1.0*g_re[0];
	g_max = g_re[1];

	timing[16] = MPI_Wtime() - timing[16];

	timing[17] = MPI_Wtime();

//	MPI_Allreduce(&min,&g_min,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
//	MPI_Allreduce(&max,&g_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

	bin_width = 2*(*E);

	// calc the number of bins given the change ratio range

	t_group_num = (int)((g_max-g_min)/bin_width);
	if((g_max-g_min)>(double)t_group_num*bin_width)
		t_group_num++;

	// malloc meta data for bins, including the bin center, number of change ratios covered by the bin and if the bin is selected as top-k bins.

	t_center = (struct t_center_element *)malloc((t_group_num+1)*sizeof(struct t_center_element));
	assert(t_center!=NULL);
	max_center = (struct t_center_element *)malloc(max_predict_bin*sizeof(struct t_center_element));

	for(i=0;i<t_group_num+1;i++)
	{
		t_center[i].center = g_min + i*bin_width + 0.5*bin_width;
		t_center[i].count = 0;
//		t_center[i].sum = 0.0;
		t_center[i].g_count = 0;
//		t_center[i].g_sum = 0.0;
		t_center[i].selected_flag = 0;
	}

	// assign each change ratio to the corresponding bin

	for(i=0;i<index;i++)
	{
		t_index = (int)((t_change_ratio_array[i] - g_min)/bin_width);
//		if(t_index==t_group_num)
//			t_index--;
		t_center[t_index].count++;
//		t_center[t_index].sum += t_change_ratio_array[i];
	}

	t_center[t_group_num-1].count += t_center[t_group_num].count;

	int *t_count,*g_t_count;

	// malloc local & global histogram

	t_count = (int *)malloc(t_group_num*sizeof(int));
	g_t_count = (int *)malloc(t_group_num*sizeof(int));

	int tt_sum = 0;
//	int gg_sum = 0;


	// build the local histogram

	for(i=0;i<t_group_num;i++)
	{
		t_count[i] = t_center[i].count;
		tt_sum += t_count[i];
	}

	if(tt_sum!=index)
	{
		printf("tt_sum = %d   index = %d\n",tt_sum,index);
		printf("errrorr!!!!!!\n");
		assert(tt_sum==index);
	}
	//local gram gen
	timing[17] = MPI_Wtime() - timing[17];

	// calc the global histogram

	timing[18] = MPI_Wtime();
	MPI_Allreduce(&(t_count[0]),&(g_t_count[0]),t_group_num,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	timing[18] = MPI_Wtime() - timing[18];

	if(DEBUG && rank==0)
	{
		/*
		FILE *fp;

		fp = fopen("histogram.txt","w");

		long t_num = 0;
		int fold = 100;

		for(i=0;i<t_group_num;i++)
		{
			if(i%fold==0)
			{
				fprintf(fp,"%ld\n",t_num);
				t_num = 0;
			}

			t_num += g_t_count[i];
//			fprintf(fp,"%d\n",g_t_count[i]);
		}
		fclose(fp);
		*/
	}

//	if(rank==0)
//		printf("reduction takes %ld\n",time_diff(t_start,t_end));

	timing[19] = MPI_Wtime();

	// exclude bins which cover no change ratios

	int non_zero_count = 0;
	for(i=0;i<t_group_num;i++)
	{
		t_center[i].g_count = g_t_count[i];
		if(g_t_count[i]!=0)
			max_center[non_zero_count++] = t_center[i];
	}

//	MPI_Barrier(MPI_COMM_WORLD);

	// select top 2, 4, 8, 16, ... bins

	int from,to;
	int l,r;
	int a,b;
	int k;
	struct t_center_element temp;
	int pivot;

	from = 0;
	to = non_zero_count-1;

	for(i=64;i>=2;i--)
	{
		if(pow(2,i)-2>non_zero_count)
			continue;

		k = pow(2,i)-2;

		while(from < to)
		{
			l = from;
			r = to;
			temp = max_center[l];
			max_center[l] = max_center[(l+r)/2];
			max_center[(l+r)/2] = temp;


			pivot = max_center[l].g_count;

			a = l;
			b = r + 1;

			while(1)
			{
				do
					++a;
				while(max_center[a].g_count>=pivot && a<= r);
				do
					--b;
				while(max_center[b].g_count < pivot && b>=l);

				if(a>=b)
					break;

				temp = max_center[a];
				max_center[a] = max_center[b];
				max_center[b] = temp;
			}

			temp = max_center[l];
			max_center[l] = max_center[b];
			max_center[b] = temp;

			if(k==b)
			{
				break;
			}
			else if(k<b)
			{
				int count = 0;

				while(1)
				{
					to = b-1;
					if((b-1>=0) && max_center[b].g_count==max_center[b-1].g_count)
						b--;
					else
						break;
				}

				if(to<k)
					break;
			}
			else
			{
				while(1)
				{
					from = b+1;
					if((b+1<=non_zero_count-1) && max_center[b].g_count==max_center[b+1].g_count)
						b++;
					else
						break;
				}

				if(from>k)
					break;

			}
		}

		from = 0;
		to = k;
	}
//	MPI_Barrier(MPI_COMM_WORLD);

//	printf("phase 2 time = %ld\n",time_diff(phase_1_time,phase_2_time));

	timing[19] = MPI_Wtime() - timing[19];

	// predict the lengths of NUMARCK files for different B

	timing[20] = MPI_Wtime();
	*B = simple_grouping_prediction(max_center,0,non_zero_count,g_zero_change_ratio_num);

	//prediction time
	timing[20] = MPI_Wtime() - timing[20];
//	MPI_Barrier(MPI_COMM_WORLD);


	timing[21] = MPI_Wtime();
	free(cluster_centroids);

	// relloc index table according to new B

	class_num = pow(2,*B);

	r_class_num = class_num - 2;
	block_size = block_length*8/(*B);

	//memory leak here !! should use relloc   :  fixed   freed several lines before
	cluster_centroids = (double *)malloc((class_num)*sizeof(double));

	int t_id;

	for(i=0;i<r_class_num;i++)
	{
		cluster_centroids[i] = max_center[i].center;

		t_id = (max_center[i].center - g_min) / bin_width;
		t_center[t_id].selected_flag = 1;
		t_center[t_id].class_id = i;
	}
	timing[21] = MPI_Wtime() - timing[21];

	// assign index 0 to a change ratio (this is not the final index), if the change ratio is not locate in top-k bins
	// assign the bin index to a change ratio, if the change ratio is covered by top-k bins 

	timing[22] = MPI_Wtime();
	for(i=0;i<index;i++)
	{

		t_id = (t_change_ratio_array[i] - g_min) / bin_width;
		if(t_change_ratio_array[i]==g_max)
			t_id--;

		if(t_center[t_id].selected_flag==1)
		{
			t_membership[i] = t_center[t_id].class_id;
		}
		else
		{
			t_membership[i] = 0;
		}
	}

//		printf("membership id generate %ld\n",time_diff(t_start,t_end));

	free(t_center);
	free(max_center);
	free(t_count);
	free(g_t_count);
	timing[22] = MPI_Wtime() - timing[22];

}

// B prediction
int simple_grouping_prediction(struct t_center_element *t_center,int total_num,int total_bin_num,int already_compressed_num)
{
	long long length;
	long long min_length;
	long long sum;
	long long compressible_data_num;
	int min_B;
	long long t_g_data_array_row_num;
	long long g_data_array_row_num;
	int i,j;

	if(DEBUG && rank==0)
		printf("total_num = %d, total_bin_num = %d already compressed_num = %d\n",total_num,total_bin_num,already_compressed_num);

	long long t_num;

	t_num = data_array_row_num;

	// get the total number of change ratios considered in top-k binning

	MPI_Allreduce(&t_num,&t_g_data_array_row_num,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
	g_data_array_row_num = t_g_data_array_row_num;

	compressible_data_num = already_compressed_num;

	min_length = pow(2,1)*sizeof(double) + (long long)(g_data_array_row_num*1/8) + (long long)(g_data_array_row_num - already_compressed_num)*sizeof(double);
	min_B = 1;

	if(DEBUG && rank==0)
		printf("B = 1 length = %ld, compressible num = %ld ,incompress = %ld index table  size =  %ld\n",min_length,already_compressed_num,(long long)(g_data_array_row_num - already_compressed_num),(long long)(g_data_array_row_num*1/8));

	// calc the smallest NUMARCK file size

	for(i=2;i<=64;i++)
	{
		if(pow(2,i)>total_bin_num)
			break;

		for(j=pow(2,i-1)-2;j<pow(2,i)-2;j++)
			compressible_data_num += t_center[j].g_count;

		length = pow(2,i)*sizeof(double) +(long)(g_data_array_row_num*i/8) + (g_data_array_row_num - compressible_data_num)*sizeof(double);

		if(DEBUG && rank==0)
		{
			printf("B = %ld length = %ld, compressible num = %ld ,incompress = %ld index table  size =  %ld\n",i,length,compressible_data_num,(g_data_array_row_num - compressible_data_num)*8,(long)(g_data_array_row_num*i/8));
		}


		if(length < min_length)
		{
			min_length = length;
			min_B = i;
		}
	}

	// this version of code does not consider B > 24

	if(min_B>24)
		min_B = 24;

	if(DEBUG && rank==0)
		printf("top-k select B = %d\n",min_B);

	return min_B;
}

// kmeans center init, select first k centers as the init center of kmeans
void cluster_initial_centroids_collection(int index,double *t_change_ratio_array)
{
	FILE *fp;
	char path[64];
	char ch[8];
	int *index_array,*index_counts,*index_displs;
	int receive_length,send_length;
	int receive_index;
	int sum;
	int i;

	index_array = (int *)malloc(size*sizeof(int));
	index_counts = (int *)malloc(size*sizeof(int));
	index_displs = (int *)malloc(size*sizeof(int));

	for(i=0;i<size;i++)
	{
		index_counts[i] = 1;
		index_displs[i] = i;
	}

	MPI_Allgatherv(&index,1,MPI_INT,index_array,index_counts,index_displs,MPI_INT,MPI_COMM_WORLD);

	if(rank==0)
	{
		receive_index = 0;

		for(i=0;i<index;i++)
		{
			if(receive_index<(class_num-2))
				cluster_centroids[receive_index++] = t_change_ratio_array[i];
			else
				break;
		}

		for(i=1;i<size;i++)
		{
			receive_length = (class_num-2) - receive_index;
			if(receive_length>index_array[i])
				receive_length = index_array[i];

			if(receive_length!=0)
				MPI_Recv((cluster_centroids+receive_index),receive_length,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

			receive_index += receive_length;
		}
	}
	else
	{
		sum = 0;
		for(i=0;i<rank;i++)
			sum += index_array[i];

		send_length = (class_num-2) - sum;

		if(send_length>0)
		{
			if(send_length>index)
				send_length = index;
			MPI_Send(t_change_ratio_array,send_length,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
		}
	}

	free(index_array);
	free(index_counts);
	free(index_displs);
}

// bits write index table blocks

void bits_packing(int *B) 
{
	long t_length;
	int t_g_index_num;
	int index_num;
	int uncompress_table_index;
	int uncompress_data_point_per_block;
	int change_ratio_index;
	int i,j;

	unsigned char * block_buff;
	long block_p;
	unsigned char temp;
	int Bp;
	int t_len;
	int helper_i;
	int k;


	change_ratio_index = 0;

	// memory allocation for all index table once.
	
	long *t_length_array;
	long total_length;

	total_length = 0;

	t_length_array = (long *)malloc(block_array_size*sizeof(long));

	for(i=0;i<block_array_size;i++)
	{
		t_length_array[i] = (long)(*B)*block_array[i].data_num/8;
		if(((*B)*block_array[i].data_num)%8!=0)
			t_length_array[i]++;

		total_length += t_length_array[i];
	}

	unsigned char *total_char_array;

	total_char_array = (unsigned char *)calloc(total_length,sizeof(unsigned char));

	unsigned char *current_buff;

	current_buff = total_char_array;

	// for fast bits operation, we eliminate 'if' clauses. helper_array_calc() calculates the bits-shift offsets for different bits operation steps

	helper_array_calc();

	// different B has different number of steps. 

	if((*B)>8&&(*B)<=16)
	{

	// iterate over all blocks

	for(i=0;i<block_array_size;i++)
	{
		t_length = t_length_array[i];
		block_buff = current_buff;
		current_buff += t_length;

		uncompress_data_point_per_block = 0;
		block_p = 0;
//		membership[0] = 128+256+1024;

		// iterate over all data points within a block

		helper_i = 0;
		for(j=0;j<block_array[i].data_num;j++)
		{
			// bits copy

			block_buff[block_p/8 + 0] |= (((unsigned char *)(membership + change_ratio_index))[0])<<helper_array[helper_i][0];
			block_buff[block_p/8 + 1] |= (((unsigned char *)(membership + change_ratio_index))[0])>>helper_array[helper_i][1];
			block_buff[block_p/8 + 1] |= (((unsigned char *)(membership + change_ratio_index))[1])<<helper_array[helper_i][2];
			block_buff[block_p/8 + 2] |= (((unsigned char *)(membership + change_ratio_index))[1])>>helper_array[helper_i][3];
			block_p += (*B);

			helper_i = (block_p) %8;

			if(membership[change_ratio_index]==class_num-1)
			{
				uncompress_data_point_per_block++;
			}
			change_ratio_index++;
		}

		// update uncompress index table pointer in block meta data

		block_array[i].uncom_buff = block_buff;

		block_array[i].uncom_buff_length = t_length;
		block_array[i].uncompress_data_point_num = uncompress_data_point_per_block;
	}

	return;
	}
	else if((*B)<=8)
	{

	// please refer B > 8 && B <= 16

	for(i=0;i<block_array_size;i++)
	{
		t_length = t_length_array[i];
		block_buff = current_buff;
		current_buff += t_length;

		uncompress_data_point_per_block = 0;
		block_p = 0;
//		membership[0] = 128+256+1024;

		helper_i = 0;
		for(j=0;j<block_array[i].data_num;j++)
		{
			block_buff[block_p/8 + 0] |= (((unsigned char *)(membership + change_ratio_index))[0])<<helper_array[helper_i][0];
			block_p += (*B);

			helper_i = (block_p) %8;

			if(membership[change_ratio_index]==class_num-1)
			{
				uncompress_data_point_per_block++;
			}
			change_ratio_index++;
		}

		block_array[i].uncom_buff = block_buff;

		block_array[i].uncom_buff_length = t_length;
		block_array[i].uncompress_data_point_num = uncompress_data_point_per_block;
	}
	}
	else if((*B)>16 && (*B)<=24)
	{

	// please refer B > 8 && B <= 16

	for(i=0;i<block_array_size;i++)
	{
		t_length = t_length_array[i];
		block_buff = current_buff;
		current_buff += t_length;

		uncompress_data_point_per_block = 0;
		block_p = 0;

		helper_i = 0;
		for(j=0;j<block_array[i].data_num;j++)
		{
			block_buff[block_p/8 + 0] |= (((unsigned char *)(membership + change_ratio_index))[0])<<helper_array[helper_i][0];
			block_buff[block_p/8 + 1] |= (((unsigned char *)(membership + change_ratio_index))[0])>>helper_array[helper_i][1];
			block_buff[block_p/8 + 1] |= (((unsigned char *)(membership + change_ratio_index))[1])<<helper_array[helper_i][2];
			block_buff[block_p/8 + 2] |= (((unsigned char *)(membership + change_ratio_index))[1])>>helper_array[helper_i][3];
			block_buff[block_p/8 + 2] |= (((unsigned char *)(membership + change_ratio_index))[2])<<helper_array[helper_i][4];
			block_buff[block_p/8 + 3] |= (((unsigned char *)(membership + change_ratio_index))[2])>>helper_array[helper_i][5];
			block_p += (*B);

			helper_i = (block_p) %8;

			if(membership[change_ratio_index]==class_num-1)
			{
				uncompress_data_point_per_block++;
			}
			change_ratio_index++;
		}

		block_array[i].uncom_buff = block_buff;

		block_array[i].uncom_buff_length = t_length;
		block_array[i].uncompress_data_point_num = uncompress_data_point_per_block;
	}
	}
}

//bits opeartion, helper array, shift offset
void helper_array_calc()
{
	int i;

	for(i=0;i<6;i++)
	{
		helper_array[i][0] = i;
		helper_array[i][1] = 8-i;
		helper_array[i][2] = i;
		helper_array[i][3] = 8-i;
		helper_array[i][4] = i;
		helper_array[i][5] = 8-i;
	}

}

//print a byte
void debug_byte(char c)
{
	int j;
//		for(j=7;j>=0;j--)
		for(j=0;j<8;j++)
		{
			if(c&(0x01<<j))
				printf("1,");
			else
				printf("0,");
		}
		printf("\n");

}

//update index after binning(combine indcies(change ratio<E) with output of binning)
void update_cluster_centroids_and_membership(double *E)
{
	int t_index;
	int i;

	for(i=class_num-3;i>=0;i--)
		cluster_centroids[i+1] = cluster_centroids[i];
	cluster_centroids[0] = 0.0;
//	cluster_centroids[class_num-1] = DBL_MAX;
	cluster_centroids[class_num-1] = 12345678910.11;

	t_index = 0;

//	for(i=0;i<data_array_row_num;i++)
//		if(membership[i]==-1)
//			membership[i] = t_membership[t_index++]+1;

	long t_count;
	t_count = 0;
	for(i=0;i<data_array_row_num;i++)
	{
		if(membership[i]==-1)
		{
			membership[i] = t_membership[t_index++]+1;
		}
		if(membership[i]>0)
		{
			if(diff_double(change_ratio_array[i],cluster_centroids[membership[i]])>(*E))
			{
				t_count ++;
				membership[i] = class_num - 1;
			}
		}
		else if(membership[i]==-2)
		{
			membership[i] = class_num - 1;
		}
	}

	// binning performance
	

	if (DEBUG) {
		long g_t_change_ratio_array_length = 0;
		long g_t_count = 0;
		long l_num;

		l_num = t_change_ratio_array_length;

		MPI_Allreduce(&l_num,&g_t_change_ratio_array_length,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(&t_count,&g_t_count,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);

		if(rank==0)
			printf("binning strategy compressed %ld out of %ld\n",g_t_change_ratio_array_length - g_t_count,g_t_change_ratio_array_length);
	}

}

// equal binning
void mpi_equal_bin(double *t_change_ratio_array,int index,int r_class_num,int *t_membership,int *B,double *E)
{
	double min,max;
	double g_min,g_max;
	double bin_width;
	int bin_pos;
	int i;

	block_size = block_length*8/(*B);

	// calc local max & min change ratios

	min = t_change_ratio_array[0];
	max = t_change_ratio_array[0];

	for(i=1;i<index;i++)
	{
		if(t_change_ratio_array[i]<min)
			min = t_change_ratio_array[i];
		if(t_change_ratio_array[i]>max)
			max = t_change_ratio_array[i];
	}

	double re[2],g_re[2];

	re[0] = -1.0*min;
	re[1] = max;

	// calc global max & min change ratios

	MPI_Allreduce(re,g_re,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

	g_min = -1.0*g_re[0];
	g_max = g_re[1];

	/*
	MPI_Allreduce(&min,&g_min,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
	MPI_Allreduce(&max,&g_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
	*/

	if(g_min<min_change_ratio)
		g_min = min_change_ratio;
	if(g_max>max_change_ratio)
		g_max = max_change_ratio;

	// equally devide the range of change ratio

	bin_width = (g_max - g_min)/(double)(r_class_num);

	for(i=0;i<r_class_num;i++)
		cluster_centroids[i] = g_min + (i + 0.5)*bin_width;

	// assign bin index

	for(i=0;i<index;i++)
	{
		bin_pos = (t_change_ratio_array[i] - g_min)/bin_width;

		if(bin_pos>(double)(r_class_num))
			bin_pos = r_class_num;
		if(bin_pos<0.0)
			bin_pos = 0;

		if(bin_pos == (double)(r_class_num))
			t_membership[i] = r_class_num - 1;
		else
			t_membership[i] = (int)floor(bin_pos);
	}
}

// log binning
void mpi_log_bin(double *t_change_ratio_array,int index,int r_class_num,int *t_membership,int *B, double *E)
{
	double min,max;
	double g_min,g_max;
	double bin_width;
	int bin_pos;
	int i;

	block_size = block_length*8/(*B);

	// calc local max & min logged change ratios

	min = log(fabs(t_change_ratio_array[0]));
	max = log(fabs(t_change_ratio_array[0]));

	for(i=1;i<index;i++)
	{
		if(log(fabs(t_change_ratio_array[i]))<min)
			min = log(fabs(t_change_ratio_array[i]));
		if(log(fabs(t_change_ratio_array[i]))>max)
			max = log(fabs(t_change_ratio_array[i]));
	}

	if(max>log(fabs(max_change_ratio)))
		max = log(fabs(max_change_ratio));

	double re[2],g_re[2];

	re[0] = -1.0*min;
	re[1] = max;

	// calc global max & min logged change ratios

	MPI_Allreduce(re,g_re,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

	g_min = -1.0*g_re[0];
	g_max = g_re[1];

	// equally devide the range of change ratio

	bin_width = (g_max - g_min)/(double)(r_class_num/2);

	for(i=0;i<r_class_num/2;i++)
		cluster_centroids[r_class_num/2 - i - 1] = exp(g_min + (i)*bin_width);

	for(i=r_class_num/2;i<r_class_num;i++)
		cluster_centroids[i] = -1.0 * cluster_centroids[r_class_num-i-1];

	// assign bin index

	double min_dist;
	double t;
	int min_bin;
	int j;
	for(i=0;i<index;i++)
	{
		min_bin = 0;
		t = t_change_ratio_array[i];
//		if(t<g_min)
//			t = g_min;
//		if(t>g_max)
//			t = g_max;

		if(t<0)
		{
			t_membership[i] = (log(fabs(t)) - g_min)/bin_width + r_class_num/2;
		}
		else
		{
			t_membership[i] = r_class_num/2 - (log(fabs(t)) - g_min)/bin_width;
		}
	}
}

//calc element wise change ratio
void calc_change_ratio(double *E)
{
	int i;

	/* membership: indices of element-wise change ratios */
	//different phase has different meaning.
	//before binning: 0:change_ratio<=E, -1: change_ratio>E & not a outlier -2: outlier_change_ratio
	//
	//after binning: this array will be updated according to the output of binning strategy(t_memership). The data will be the integer index of data points
	membership = (int *)malloc(data_array_row_num*sizeof(int));
	assert(membership!=NULL);

	/* t_membership: help array for membership to be used in binning phase */
	//output of binning strategy
	
	t_membership = (int *)malloc(data_array_row_num*sizeof(int));
	assert(t_membership!=NULL);

	/* change_ratio_array: element-wise change ratios */
	change_ratio_array = (double *)malloc(data_array_row_num*sizeof(double));
	assert(change_ratio_array!=NULL);

//	change_ratio_flag_array = (int *)malloc(data_array_row_num*sizeof(int));
//	assert(change_ratio_flag_array!=NULL);

	int t_sum, g_t_sum;

	zero_change_ratio_num = 0;

	int t_count = 0;
	for(i=0;i<data_array_row_num;i++) /* for each element */
	{
		if(data_array2[i]==data_array1[i])
			t_count ++;
		if(data_array1[i]==0.0)  /* zero-value element */
		{

			if(data_array2[i]==0.0)
			{
				change_ratio_array[i] = 0.0;
				zero_change_ratio_num ++;
				membership[i] = 0;
			}
			else /* data_array2[i] != 0.0 */
			{
				change_ratio_array[i] = max_change_ratio;
				membership[i] = -2;
			}
		}
		else /* data_array1[i] != 0.0 */
		{
			change_ratio_array[i] = (data_array2[i] - data_array1[i])/(data_array1[i]);

			if(change_ratio_array[i]>max_change_ratio||change_ratio_array[i]<min_change_ratio)
			{
				t_sum ++;
				membership[i] = -2;
			}
			else
			{
				if(diff_double(change_ratio_array[i],0.0)<=(*E))
				{
					zero_change_ratio_num ++;
					membership[i] = 0;
				}
				else
				{
					membership[i] = -1;
				}

			}
		}
	}

	free(data_array1);

	// t_change_ratio_array: helper for change_ratio_array
	// this helper array saves the input change ratios for binning (change_ratios<=E & outlier_change_ratios are excluded)


	t_change_ratio_array = (double *)malloc(data_array_row_num*sizeof(double));
	assert(t_change_ratio_array!=NULL);
	t_change_ratio_array_length = 0;

	for(i=0;i<data_array_row_num;i++)
	{
		if(membership[i]==-1)
			t_change_ratio_array[t_change_ratio_array_length++] = change_ratio_array[i];
	}

}

#ifdef _ADD_HDF5_
//read hdf5 input file
void read_hdf5_input_file()
{
	if(float_flag==1)
		read_hdf5_float_input_file();
	if(float_flag==0)
		read_hdf5_double_input_file();
}

//read float hdf5
//please refer the comments in load_netcdf_file_all_float()
void read_hdf5_float_input_file()
{
	FILE *fp;
	float *buff;
	char path1[128];
	char path2[128];
	char ch[5];
	int ndims;
	hsize_t dims[10],memdims[10],count[10],offset[10],stride[10],block[10];
	long size_of_data;
	int start_1D_id,end_1D_id;
	long i;
	hid_t file_id,dataset_id,dataspace_id,memspace_id;

	int *map_array;

	size_of_data = 0;

	strcpy(path1,input_file_1);


	file_id = H5Fopen (path1, H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset_id = H5Dopen (file_id, var_name, H5P_DEFAULT);
	dataspace_id = H5Dget_space(dataset_id);

	ndims = H5Sget_simple_extent_ndims(dataspace_id);
	assert(ndims<=10);
	H5Sget_simple_extent_dims(dataspace_id,dims,NULL);

	H5Sclose(dataspace_id);
	H5Dclose(dataset_id);
	H5Fclose(file_id);

	size_of_data = 1;
	for(i=0;i<ndims;i++)
	{
		size_of_data *= dims[i];
	}

	map_array = (int *)malloc(size*sizeof(int));

	for(i=0;i<size-1;i++)
		map_array[i] = size_of_data/size;

	map_array[size-1] = size_of_data - (size - 1)*(size_of_data/size);

	g_block_map = (struct g_block_map_element *)malloc(size*sizeof(struct g_block_map_element));

	long t_start_index;
	t_start_index = 0;
	for(i=0;i<size;i++)
	{
		g_block_map[i].unit_num = map_array[i];
		g_block_map[i].start_index = t_start_index;
		g_block_map[i].end_index = t_start_index+map_array[i]-1;
		g_block_map[i].real_start_index = t_start_index;
		g_block_map[i].real_end_index = t_start_index+map_array[i]-1;
		t_start_index += map_array[i];
		g_block_map[i].send_r_id = -1;
		g_block_map[i].send_r_num = 0;
		g_block_map[i].send_l_id = -1;
		g_block_map[i].send_l_num = 0;
	}


	data_array_row_num = map_array[rank];

	data_array1 = (double *)malloc(data_array_row_num*sizeof(double));
	data_array2 = (double *)malloc(data_array_row_num*sizeof(double));

	long rest_dims = 1;

	for(i=1;i<ndims;i++)
		rest_dims *= dims[i];

	start_1D_id = g_block_map[rank].start_index/(rest_dims);
	end_1D_id = (g_block_map[rank].start_index+data_array_row_num-1)/(rest_dims);

	offset[0] = start_1D_id;
	for(i=1;i<ndims;i++)
		offset[i] = 0;

	count[0] = end_1D_id - start_1D_id + 1;
	for(i=1;i<ndims;i++)
		count[i] = dims[i];

	for(i=0;i<ndims;i++)
		stride[i] = 1;

	for(i=0;i<ndims;i++)
		block[i] = 1;

	for(i=0;i<ndims;i++)
		memdims[i] = count[i];

	long count_total;
	count_total = 1;

	for(i=0;i<ndims;i++)
		count_total *= count[i];

	buff = (float *)malloc(count_total*sizeof(float));

	strcpy(path1,input_file_1);

	hid_t plist_id;

	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	assert(plist_id>=0);
	H5Pset_fapl_mpio(plist_id,MPI_COMM_WORLD,MPI_INFO_NULL);

	file_id = H5Fopen (path1, H5F_ACC_RDONLY, plist_id);
	H5Pclose(plist_id);
	dataset_id = H5Dopen (file_id, var_name, H5P_DEFAULT);
	dataspace_id = H5Dget_space(dataset_id);

	memspace_id = H5Screate_simple(ndims,memdims,NULL);

	for(i=0;i<ndims;i++)
		assert(count[i]>0);

//	MPI_Barrier(MPI_COMM_WORLD);
//	H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,stride,count,block);
	H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,NULL,count,NULL);

//	hid_t plist_id;
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	H5Dread(dataset_id,H5T_IEEE_F32LE,memspace_id,dataspace_id,plist_id,buff);
//	H5Dread(dataset_id,H5T_IEEE_F32LE,memspace_id,dataspace_id,H5P_DEFAULT,buff);
	H5Dclose(dataset_id);
	H5Pclose(plist_id);
	H5Sclose(memspace_id);
	H5Sclose(dataspace_id);
	H5Fclose(file_id);

	long start_index;
	long t_i;
	long rest_count;

	rest_count = 1;

	for(i=1;i<ndims;i++)
		rest_count *= count[i];
	start_index = start_1D_id*rest_count;

	t_i = 0;
	for(i=0;i<count[0]*rest_count;i++)
	{
		if(start_index+i>=g_block_map[rank].start_index&&(start_index+i)<(g_block_map[rank].start_index+data_array_row_num))
			data_array1[t_i++] = buff[i];
	}

	strcpy(path2,input_file_2);

	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	assert(plist_id>=0);
	H5Pset_fapl_mpio(plist_id,MPI_COMM_WORLD,MPI_INFO_NULL);

	file_id = H5Fopen (path2, H5F_ACC_RDONLY, plist_id);
	assert(file_id!=-1);
	H5Pclose(plist_id);
	dataset_id = H5Dopen (file_id, var_name, H5P_DEFAULT);
	assert(dataset_id!=-1);
	dataspace_id = H5Dget_space(dataset_id);
	assert(dataspace_id!=-1);

	memspace_id = H5Screate_simple(ndims,memdims,NULL);
	assert(memspace_id!=-1);

//	H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,stride,count,block);
	H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,NULL,count,NULL);
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
	H5Dread(dataset_id,H5T_IEEE_F32LE,memspace_id,dataspace_id,plist_id,buff);
//	H5Dread(dataset_id,H5T_IEEE_F32LE,memspace_id,dataspace_id,H5P_DEFAULT,buff);
	H5Dclose(dataset_id);
//	H5Pclose(plist_id);
	H5Sclose(memspace_id);
	H5Sclose(dataspace_id);
	H5Fclose(file_id);

	start_index = start_1D_id*rest_count;

	t_i = 0;
	for(i=0;i<count[0]*rest_count;i++)
	{
		if(start_index+i>=g_block_map[rank].start_index&&(start_index+i)<(g_block_map[rank].start_index+data_array_row_num))
			data_array2[t_i++] = buff[i];
	}

	free(buff);
}

//read double hdf5
//please refer the comments in load_netcdf_file_all_float()
void read_hdf5_double_input_file()
{
	FILE *fp;
	double *buff;
	char path1[128];
	char path2[128];
	char ch[5];
	int ndims;
	hsize_t dims[10],memdims[10],count[10],offset[10],stride[10],block[10];
	long size_of_data;
	int start_1D_id,end_1D_id;
	long i;
	hid_t file_id,dataset_id,dataspace_id,memspace_id;

	int *map_array;

	size_of_data = 0;

	strcpy(path1,input_file_1);

	file_id = H5Fopen (path1, H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset_id = H5Dopen (file_id, var_name, H5P_DEFAULT);
	dataspace_id = H5Dget_space(dataset_id);

	ndims = H5Sget_simple_extent_ndims(dataspace_id);
	assert(ndims<=10);
	H5Sget_simple_extent_dims(dataspace_id,dims,NULL);

	H5Sclose(dataspace_id);
	H5Dclose(dataset_id);
	H5Fclose(file_id);

	size_of_data = 1;
	for(i=0;i<ndims;i++)
	{
		size_of_data *= dims[i];
	}

	map_array = (int *)malloc(size*sizeof(int));

	for(i=0;i<size-1;i++)
		map_array[i] = size_of_data/size;

	map_array[size-1] = size_of_data - (size - 1)*(size_of_data/size);

	g_block_map = (struct g_block_map_element *)malloc(size*sizeof(struct g_block_map_element));

	long t_start_index;
	t_start_index = 0;
	for(i=0;i<size;i++)
	{
		g_block_map[i].unit_num = map_array[i];
		g_block_map[i].start_index = t_start_index;
		g_block_map[i].end_index = t_start_index+map_array[i]-1;
		g_block_map[i].real_start_index = t_start_index;
		g_block_map[i].real_end_index = t_start_index+map_array[i]-1;
		t_start_index += map_array[i];
		g_block_map[i].send_r_id = -1;
		g_block_map[i].send_r_num = 0;
		g_block_map[i].send_l_id = -1;
		g_block_map[i].send_l_num = 0;
	}


	data_array_row_num = map_array[rank];

	data_array1 = (double *)malloc(data_array_row_num*sizeof(double));
	data_array2 = (double *)malloc(data_array_row_num*sizeof(double));

	long rest_dims = 1;

	for(i=1;i<ndims;i++)
		rest_dims *= dims[i];

	start_1D_id = g_block_map[rank].start_index/(rest_dims);
	end_1D_id = (g_block_map[rank].start_index+data_array_row_num-1)/(rest_dims);

	offset[0] = start_1D_id;
	for(i=1;i<ndims;i++)
		offset[i] = 0;

	count[0] = end_1D_id - start_1D_id + 1;
	for(i=1;i<ndims;i++)
		count[i] = dims[i];

	for(i=0;i<ndims;i++)
		stride[i] = 1;

	for(i=0;i<ndims;i++)
		block[i] = 1;

	for(i=0;i<ndims;i++)
		memdims[i] = count[i];

	long count_total;
	count_total = 1;

	for(i=0;i<ndims;i++)
		count_total *= count[i];

	buff = (double *)malloc(count_total*sizeof(double));

	strcpy(path1,input_file_1);

	file_id = H5Fopen (path1, H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset_id = H5Dopen (file_id, var_name, H5P_DEFAULT);
	dataspace_id = H5Dget_space(dataset_id);

	memspace_id = H5Screate_simple(ndims,memdims,NULL);

	for(i=0;i<ndims;i++)
		assert(count[i]>0);

//	MPI_Barrier(MPI_COMM_WORLD);
	H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,stride,count,block);
	H5Dread(dataset_id,H5T_NATIVE_DOUBLE,memspace_id,dataspace_id,H5P_DEFAULT,buff);
	H5Dclose(dataset_id);
	H5Sclose(memspace_id);
	H5Sclose(dataspace_id);
	H5Fclose(file_id);

	long start_index;
	long t_i;
	long rest_count;

	rest_count = 1;

	for(i=1;i<ndims;i++)
		rest_count *= count[i];
	start_index = start_1D_id*rest_count;

	t_i = 0;
	for(i=0;i<count[0]*rest_count;i++)
	{
		if(start_index+i>=g_block_map[rank].start_index&&(start_index+i)<(g_block_map[rank].start_index+data_array_row_num))
			data_array1[t_i++] = buff[i];
	}

	strcpy(path2,input_file_2);

	file_id = H5Fopen (path2, H5F_ACC_RDONLY, H5P_DEFAULT);
	assert(file_id!=-1);
	dataset_id = H5Dopen (file_id, var_name, H5P_DEFAULT);
	assert(dataset_id!=-1);
	dataspace_id = H5Dget_space(dataset_id);
	assert(dataspace_id!=-1);

	memspace_id = H5Screate_simple(ndims,memdims,NULL);
	assert(memspace_id!=-1);

	H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,stride,count,block);
	H5Dread(dataset_id,H5T_NATIVE_DOUBLE,memspace_id,dataspace_id,H5P_DEFAULT,buff);
	H5Dclose(dataset_id);
	H5Sclose(memspace_id);
	H5Sclose(dataspace_id);
	H5Fclose(file_id);

	start_index = start_1D_id*rest_count;

	t_i = 0;
	for(i=0;i<count[0]*rest_count;i++)
	{
		if(start_index+i>=g_block_map[rank].start_index&&(start_index+i)<(g_block_map[rank].start_index+data_array_row_num))
			data_array2[t_i++] = buff[i];
	}

	free(buff);

}
#endif

#ifdef _ADD_PNETCDF_
//read pnetcdf input file
void read_pnetcdf_input_file()
{
	if(float_flag==1)
	{
		load_netcdf_file_all_float(input_file_1,&data_array1);
		load_netcdf_file_all_float(input_file_2,&data_array2);
	}
	else
	{
		load_netcdf_file_all_double(input_file_1,&data_array1);
		load_netcdf_file_all_double(input_file_2,&data_array2);
	}
}
#endif

#ifdef _ADD_PNETCDF_
//read float pnetcdf
void load_netcdf_file_all_float(char *path,double **array)
{

	MPI_Offset *dim_sizes, var_size,t_var_size,total_var_size;
	MPI_Offset *start, *count;
	float *buff;
	nc_type type;
	int dimids[NC_MAX_VAR_DIMS];
	int ncfile, ndims, nvars, ngatts, unlimited;
	int var_ndims, var_natts;;
	int block_num;
	int t_offset;
	int g_start,g_length;
	int ret;
	long t_long;
	long i,j,k;

	ret = ncmpi_open(MPI_COMM_WORLD, path, NC_NOWRITE, MPI_INFO_NULL,&ncfile);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_inq(ncfile, &ndims, &nvars, &ngatts, &unlimited);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	dim_sizes = (MPI_Offset*) calloc(ndims, sizeof(MPI_Offset));

	for(i=0; i<ndims; i++)
	{
		ret = ncmpi_inq_dimlen(ncfile, i, &(dim_sizes[i]) );
		if (ret != NC_NOERR) handle_error(ret, __LINE__);
	}

	start = (MPI_Offset*) calloc(100, sizeof(MPI_Offset));
	count = (MPI_Offset*) calloc(100, sizeof(MPI_Offset));

	var_size = 0;

	long size_of_data;

	ncmpi_inq_varid(ncfile,var_name,&var_index);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_inq_var(ncfile, var_index, var_name, &type, &var_ndims, dimids,&var_natts);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	// calculate total size of the input variable

	size_of_data = 1;
	for(i=0;i<var_ndims;i++)
		size_of_data *=dim_sizes[dimids[i]];

	int *map_array;

	map_array = (int *)malloc(size*sizeof(int));

	// devide the variable into chunkes
	// map_array indicates the size of the variable of each MPI process
	for(i=0;i<size-1;i++)
		map_array[i] = size_of_data/size;

	map_array[size-1] = size_of_data - (size - 1)*(size_of_data/size);

	// init the meta data (g_block_map), it contains meta information for data assigned to a mpi process, including offsets, length, the number of data points exchanged with neighbors, which will be init later.
	
	g_block_map = (struct g_block_map_element *)malloc(size*sizeof(struct g_block_map_element));

	long t_start_index;
	t_start_index = 0;
	for(i=0;i<size;i++)
	{
		g_block_map[i].unit_num = map_array[i];
		g_block_map[i].start_index = t_start_index;
		g_block_map[i].end_index = t_start_index+map_array[i]-1;
		g_block_map[i].real_start_index = t_start_index;
		g_block_map[i].real_end_index = t_start_index+map_array[i]-1;
		t_start_index += map_array[i];
		g_block_map[i].send_r_id = -1;
		g_block_map[i].send_r_num = 0;
		g_block_map[i].send_l_id = -1;
		g_block_map[i].send_l_num = 0;
	}

	// malloc space for local data
	data_array_row_num = map_array[rank];

	*array = (double *)malloc(data_array_row_num*sizeof(double));

	long rest_dims = 1;

	for(i=1;i<var_ndims;i++)
		rest_dims *=dim_sizes[dimids[i]];

	int start_1D_id;
	int end_1D_id;

	// calculate the offsets & length of a chunk(sub-array) of data which covers the range of local data. 
	// note that local data may not be a sub-array of data. We need to select a sub-array which covers the local data.
	
	start_1D_id = g_block_map[rank].start_index/(rest_dims);
	end_1D_id = (g_block_map[rank].start_index+data_array_row_num-1)/(rest_dims);

	start[0] = start_1D_id;
	for(i=1;i<var_ndims;i++)
		start[i] = 0;

	count[0] = end_1D_id - start_1D_id + 1;
	for(i=1;i<var_ndims;i++)
		count[i] = dim_sizes[dimids[i]];

	long count_total;

	count_total = 1;
	for(i=0;i<var_ndims;i++)
		count_total *= count[i];

	// read data into a temporal buffer

	buff = (float *) calloc(count_total, sizeof(float));
	ret = ncmpi_get_vara_float_all(ncfile, var_index, start, count, buff);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	long t_i;

	long rest_count;

	rest_count = 1;
	for(i=1;i<var_ndims;i++)
		rest_count *= count[i];

	long start_index;

	// copy the corresponding part to local data.

	start_index = start_1D_id*rest_count;
	t_i = 0;
	for(i=0;i<count[0]*rest_count;i++)
	{
		if(start_index+i>=g_block_map[rank].start_index&&(start_index+i)<(g_block_map[rank].start_index+data_array_row_num))
			(*array)[t_i++] = buff[i];
	}

	ret = ncmpi_close(ncfile);
	free(map_array);
	free(buff);

}

//read double pnetcdf. 
//please refer the comments in load_netcdf_file_all_float()
void load_netcdf_file_all_double(char *path,double **array)
{

	MPI_Offset *dim_sizes, var_size,t_var_size,total_var_size;
	MPI_Offset *start, *count;
	double *buff;
	nc_type type;
	int dimids[NC_MAX_VAR_DIMS];
	int ncfile, ndims, nvars, ngatts, unlimited;
	int var_ndims, var_natts;;
	int block_num;
	int t_offset;
	int g_start,g_length;
	int ret;
	long t_long;
	long i,j,k;

	ret = ncmpi_open(MPI_COMM_WORLD, path, NC_NOWRITE, MPI_INFO_NULL,&ncfile);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_inq(ncfile, &ndims, &nvars, &ngatts, &unlimited);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	dim_sizes = (MPI_Offset*) calloc(ndims, sizeof(MPI_Offset));

	for(i=0; i<ndims; i++)
	{
		ret = ncmpi_inq_dimlen(ncfile, i, &(dim_sizes[i]) );
		if (ret != NC_NOERR) handle_error(ret, __LINE__);
	}

	start = (MPI_Offset*) calloc(100, sizeof(MPI_Offset));
	count = (MPI_Offset*) calloc(100, sizeof(MPI_Offset));

	var_size = 0;

	long size_of_data;


	for(k=0;k<1;k++)
	{
		ncmpi_inq_varid(ncfile,var_name,&var_index);
		if (ret != NC_NOERR) handle_error(ret, __LINE__);

		ret = ncmpi_inq_var(ncfile, var_index, var_name, &type, &var_ndims, dimids,&var_natts);
		if (ret != NC_NOERR) handle_error(ret, __LINE__);

		size_of_data = 1;
		for(i=0;i<var_ndims;i++)
			size_of_data *=dim_sizes[dimids[i]];
	}

	int *map_array;

	map_array = (int *)malloc(size*sizeof(int));

	for(i=0;i<size-1;i++)
		map_array[i] = size_of_data/size;

	map_array[size-1] = size_of_data - (size - 1)*(size_of_data/size);

	g_block_map = (struct g_block_map_element *)malloc(size*sizeof(struct g_block_map_element));

	long t_start_index;
	t_start_index = 0;
	for(i=0;i<size;i++)
	{
		g_block_map[i].unit_num = map_array[i];
		g_block_map[i].start_index = t_start_index;
		g_block_map[i].end_index = t_start_index+map_array[i]-1;
		g_block_map[i].real_start_index = t_start_index;
		g_block_map[i].real_end_index = t_start_index+map_array[i]-1;
		t_start_index += map_array[i];
		g_block_map[i].send_r_id = -1;
		g_block_map[i].send_r_num = 0;
		g_block_map[i].send_l_id = -1;
		g_block_map[i].send_l_num = 0;
	}

	data_array_row_num = map_array[rank];

	*array = (double *)malloc(data_array_row_num*sizeof(double));

	long rest_dims = 1;

	for(i=1;i<var_ndims;i++)
		rest_dims *=dim_sizes[dimids[i]];

	int start_1D_id;
	int end_1D_id;

	start_1D_id = g_block_map[rank].start_index/(rest_dims);
	end_1D_id = (g_block_map[rank].start_index+data_array_row_num-1)/(rest_dims);

	start[0] = start_1D_id;
	for(i=1;i<var_ndims;i++)
		start[i] = 0;

	count[0] = end_1D_id - start_1D_id + 1;
	for(i=1;i<var_ndims;i++)
		count[i] = dim_sizes[dimids[i]];

	long count_total;

	count_total = 1;
	for(i=0;i<var_ndims;i++)
		count_total *= count[i];

	buff = (double *) calloc(count_total, sizeof(double));
	ret = ncmpi_get_vara_double_all(ncfile, var_index, start, count, buff);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	long t_i;

	long rest_count;

	rest_count = 1;
	for(i=1;i<var_ndims;i++)
		rest_count *= count[i];

	long start_index;

	start_index = start_1D_id*rest_count;
	t_i = 0;
	for(i=0;i<count[0]*rest_count;i++)
	{
		if(start_index+i>=g_block_map[rank].start_index&&(start_index+i)<(g_block_map[rank].start_index+data_array_row_num))
			(*array)[t_i++] = buff[i];
	}

	ret = ncmpi_close(ncfile);
	free(map_array);
	free(buff);

}

#endif

//update how many data points need to be exchanged with its neighbors
void g_block_adjust(int *B)
{
	int t_block_size;
	int t_num;
	int c_num;
	long i;

	t_block_size = block_length*8/(*B);

	for(i=0;i<size-1;i++)
	{
		t_num = g_block_map[i].real_end_index - g_block_map[i].real_start_index + 1;

		c_num = t_num%t_block_size;

		g_block_map[i].real_end_index -= c_num;

		g_block_map[i+1].real_start_index -= c_num;
	}

	for(i=0;i<size;i++)
	{
		if(g_block_map[i].real_end_index<g_block_map[i].real_start_index&&rank==0)
			printf("some MPI process do not have enough data to compress, please use less MPI processes || higher B || smaller block size\n");
		assert(g_block_map[i].real_end_index>=g_block_map[i].real_start_index);
	}
}

void change_ratio_transfer(int *index_num) /* exchange index with neighbors */
{
	MPI_Request l_request,r_request;
	struct timeval t_start,t_end;
	unsigned char *ret;
	unsigned char *seg,*body;
	unsigned char *t_buff;
	unsigned char *recv_buff_l,*local_body,*recv_buff_r;
	int *l_send_buff,*r_send_buff;
	int *recv_l_buff,*recv_r_buff;
	int recv_r_index,recv_l_index;
	long sended_num,received_l_num,received_r_num;
	long received_num;
	long recv_buff_l_length,body_length,recv_buff_r_length;
	long t_length;
	long buff_length;
	int i;

	long t_mpi_time = 0;
	long g_t_mpi_time = 0;
	received_num = 0;
	received_l_num = 0;
	received_r_num = 0;

	for(i=0;i<size;i++)
	{
		if(g_block_map[i].send_r_id==rank)
		{
			received_num += g_block_map[i].send_r_num;
			received_l_num += g_block_map[i].send_r_num;
		}

		if(g_block_map[i].send_l_id==rank)
		{
			received_num += g_block_map[i].send_l_num;
			received_r_num += g_block_map[i].send_l_num;
		}

	}

		for(i=0;i<size;i++)
		{
			assert(g_block_map[i].send_l_num>=0);
			assert(g_block_map[i].send_r_num>=0);
			assert(received_num>=0);
			assert(received_l_num>=0);
			assert(received_r_num>=0);
		}

	sended_num = 0;
	sended_num += g_block_map[rank].send_r_num;
	sended_num += g_block_map[rank].send_l_num;

	*index_num = g_block_map[rank].unit_num+received_num-sended_num;

	if(g_block_map[rank].send_l_id!=-1)
	{
		l_send_buff = (int *)malloc(g_block_map[rank].send_l_num*sizeof(int));
		assert(l_send_buff!=NULL);
		for(i=0;i<g_block_map[rank].send_l_num;i++)
		{
			l_send_buff[i] = membership[i];
		}

		MPI_Isend(l_send_buff,g_block_map[rank].send_l_num,MPI_INT,g_block_map[rank].send_l_id,0,MPI_COMM_WORLD,&l_request);

	}

	if(g_block_map[rank].send_r_id!=-1)
	{
		r_send_buff = (int *)malloc(g_block_map[rank].send_r_num*sizeof(int));
		assert(r_send_buff!=NULL);
		for(i=0;i<g_block_map[rank].send_r_num;i++)
		{
			r_send_buff[i] = membership[data_array_row_num-g_block_map[rank].send_r_num+i];
		}

		MPI_Isend(r_send_buff,g_block_map[rank].send_r_num,MPI_INT,g_block_map[rank].send_r_id,0,MPI_COMM_WORLD,&r_request);
	}

	recv_r_buff = (int *)malloc(received_r_num*sizeof(int));
	assert(recv_r_buff!=NULL);
	recv_r_index = 0;
	for(i=0;i<size;i++)
	{
		if(g_block_map[i].send_l_id==rank)
		{
			MPI_Recv(recv_r_buff+recv_r_index,g_block_map[i].send_l_num,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			recv_r_index += g_block_map[i].send_l_num;
		}
	}

	recv_l_buff = (int *)malloc(received_l_num*sizeof(int));
	assert(recv_l_buff!=NULL);
	recv_l_index = 0;
	for(i=0;i<size;i++)
	{
		if(g_block_map[i].send_r_id==rank)
		{
			MPI_Recv(recv_l_buff+recv_l_index,g_block_map[i].send_r_num,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			recv_l_index += g_block_map[i].send_r_num;
		}
	}

	if(g_block_map[rank].send_l_id!=-1)
	{
		MPI_Wait(&l_request,MPI_STATUS_IGNORE);
		free(l_send_buff);
	}

	if(g_block_map[rank].send_r_id!=-1)
	{
		MPI_Wait(&r_request,MPI_STATUS_IGNORE);
		free(r_send_buff);
	}

	membership = (int *)realloc(membership,*index_num*sizeof(int));
	assert(membership!=NULL);

	for(i=data_array_row_num-1-g_block_map[rank].send_r_num;i>=0+g_block_map[rank].send_l_num;i--)
	{
		membership[i+received_l_num] = membership[i];
	}

	for(i=0;i<received_l_num;i++)
	{
		membership[i] = recv_l_buff[i];
	}

	for(i=0;i<received_r_num;i++)
	{
		membership[i+data_array_row_num+received_l_num] = recv_r_buff[i];
	}

	free(recv_r_buff);
	free(recv_l_buff);

}

/* set MPI send ID & send num according to the result of g_block_adjust */

void data_split()
{
	int i;
	for(i=0;i<size-1;i++)
	{
		if(g_block_map[i].real_end_index>g_block_map[i].end_index)
		{
			g_block_map[i+1].send_l_id = i;
			g_block_map[i+1].send_l_num = g_block_map[i].real_end_index - g_block_map[i].end_index;
		}
		else if(g_block_map[i].real_end_index < g_block_map[i].end_index)
		{
			g_block_map[i].send_r_id = i+1;
			g_block_map[i].send_r_num = g_block_map[i].end_index - g_block_map[i].real_end_index;
		}
	}
}

int calc_send_num(int b_index,int s_index,int e_index)
{
	int max_s,min_e;

	max_s = g_block_map[b_index].start_index;
	if(max_s<s_index)
		max_s = s_index;

	min_e = g_block_map[b_index].end_index;
	if(min_e>e_index)
		min_e = e_index;

	if(min_e<max_s)
	{
//		printf("error here! min_e = %d   max_s = %d  b_index = %d , s_index = %d, e_index = %d rank = %d\n",min_e,max_s,b_index,g_block_map[b_index].start_index,g_block_map[b_index].end_index,rank);
//		if(g_block_map[])
//		assert(min_e>=max_s);
	}

	return min_e-max_s+1;
}

int left_size(int index)
{
	int block_id;

	block_id = block_index(index);

	return g_block_map[block_id].end_index-index+1;
}

int right_size(int index)
{
	int block_id;

	block_id = block_index(index);

	return index-g_block_map[block_id].start_index+1;
}

int block_index(int index)
{
	int i;

	for(i=0;i<size;i++)
	{
		if((index>=g_block_map[i].start_index)&&(index<=g_block_map[i].end_index))
			return i;
	}

	printf("block_index fail! index %d cannot find!\n",index);
	exit(1);
}

int calc_mini_block_num(int start,int length,int dim0_size)
{
	if((start+length)*dim0_block_size<=dim0_size)
		return length*dim0_block_size;
	else
		return (dim0_size - start*dim0_block_size);
}

long int time_diff(struct timeval t_s,struct timeval t_e)
{
	long int ret;

	ret = t_e.tv_sec - t_s.tv_sec;

	ret = ret*1000000;
	ret = ret + (t_e.tv_usec - t_s.tv_usec);

	if(ret<0)
	{
		printf("time calc error!");
		exit(0);
	}
//	return ret/1000;
	return ret;
}

#ifdef _ADD_PNETCDF_
static void handle_error(int status, int lineno)
{
	fprintf(stderr, "Error at line %d: %s\n", lineno, ncmpi_strerror(status));
	MPI_Abort(MPI_COMM_WORLD, 1);
}
#endif

void itoa(char *ch,int num)
{
	ch[0] = '0' + (num/100)%10;
	ch[1] = '0' + (num/10)%10;
	ch[2] = '0' + num%10;
	ch[3] = '\0';
}
