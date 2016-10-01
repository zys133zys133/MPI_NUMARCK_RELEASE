#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#ifdef _ADD_PNETCDF_
#include "pnetcdf.h"
#endif
#ifdef _ADD_HDF5_
#include "hdf5.h"
#endif
#include "zlib.h"
#include "assert.h"
#include "time.h"


#define BLOCK_LOW(id, p, n) ((id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1, p, n)-1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1)
#define BLOCK_OWNER(index, p, n) (((P) * ((index) +1)-1)/(n))
#define ERR(e) {printf("Error: %s\n", nc_strerror(e));}

int *offset_block_length;
int *offset_incompressible;
int random_start,random_end;
double *data_array;			//change
double *ori_data_array;
double *cmp_data_array;
int * index_id;
int block_size;

int var_index;
char *var_name = NULL;

char *ori_file = NULL;
char *cmp_file = NULL;
char *numarck_file = NULL;
int input_format = 0;
int output_format = 0;

int dim_nxb = 32;
int dim_nyb = 32;

int float_flag = 1;

int rank,size;
int incompressible_data_num;
int B;
int block_array_size;
int helper_array[8][8];
unsigned char helper_array_mask[8][8];

int data_array_row_num;
int ori_data_array_row_num;
int class_num;

void read_ori_file(int file_index);
void load_netcdf_file_all_float(char *path,double **array);
void load_netcdf_file_all_double(char *path,double **array);
static void handle_error(int status, int lineno);
void itoa(char *ch,int num);
void random_de_comp_double();
void random_de_comp_float();
int block_overlap(int a,int b,int c,int d);
void helper_array_calc_decomp();
unsigned char mask_gen(int r_length,int offset,int flag);
long int time_diff(struct timeval t_s,struct timeval t_e);
float diff(float a,float b);
void debug_byte(char c);
void read_pnetcdf_input_file(int from);
void read_hdf5_input_file(int from);
void cmd_helper();
void read_hdf5_float_input_file(char *path,double **data_array);
void read_hdf5_double_input_file(char *path,double **data_array);

int main(int argc,char *argv[])
{

	int file_id;
	int i;

	extern char * optarg;
	int c;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	random_start = -1;
	random_end = -1;

	if(size!=1)
	{
		if(rank==0)
			printf("NUMARCK does not support parallel decompress in this version. Please use only one MPI process.\n");
		MPI_Finalize();

		return;
	}

	while((c = getopt(argc,argv,"i:j:o:v:s:e:hdxy"))!=-1)
	{
		switch(c)
		{
			case 'i':
				ori_file = strdup(optarg);
				break;
			case 'j':
				cmp_file = strdup(optarg);
				break;
			case 'o':
				numarck_file = strdup(optarg);
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
				printf("not support hdf5 NUMARCk in this version\n");
				break;
			case 's':
				random_start = atoi(optarg);
				break;
			case 'e':
				random_end = atoi(optarg);
				break;
			case 'h':
				cmd_helper();
				MPI_Finalize();
				return;
				break;
		}
	}

	if(ori_file==NULL)
	{
		printf("ori file empty!\n");
		exit(1);
	}

	if(cmp_file==NULL)
	{
		printf("cmp file empty!\n");
		exit(1);
	}

	if(numarck_file==NULL)
	{
		printf("NUMARCK file empty!\n");
		exit(1);
	}

	if(var_name==NULL)
	{
		printf("var name empty!\n");
		exit(1);
	}


#ifdef _ADD_PNETCDF_
	if(input_format==0)
	{
		read_pnetcdf_input_file(0);
	}
#endif

#ifdef _ADD_HDF5_
	if(input_format==1)
	{
		read_hdf5_input_file(0);
	}
#endif

	if(random_start==-1&&random_end==-1)
	{
		random_start = 0;
		random_end = data_array_row_num -1;
	}

	data_array = (double *)malloc((random_end-random_start+1)*sizeof(double));
	index_id = (int *)malloc((random_end-random_start+1)*sizeof(int));

#ifdef _ADD_PNETCDF_

	if(float_flag==1)
		random_de_comp_float();
	else
		random_de_comp_double();
#endif

#ifdef _ADD_PNETCDF_
	if(input_format==0)
	{
		read_pnetcdf_input_file(1);
	}
#endif

#ifdef _ADD_HDF5_
	if(input_format==1)
	{
		read_hdf5_input_file(1);
	}
#endif


	float sum;

	for(i=random_start;i<random_end;i++)
	{
		sum += diff(data_array[i],ori_data_array[i]);
	}

	printf("ave_abs_error = %f\n",sum/(float)(random_end-random_start));

	MPI_Finalize();

	return;

}

#ifdef _ADD_HDF5_
//read hdf5 input file
void read_hdf5_input_file(int from)
{
	if(from==0)
	{
		if(float_flag==1)
			read_hdf5_float_input_file(ori_file,&ori_data_array);
		if(float_flag==0)
			read_hdf5_double_input_file(ori_file,&ori_data_array);
	}
	else
	{
		if(float_flag==1)
			read_hdf5_float_input_file(cmp_file,&cmp_data_array);
		if(float_flag==0)
			read_hdf5_double_input_file(cmp_file,&cmp_data_array);
	}
}

//read float hdf5
void read_hdf5_float_input_file(char *path,double **data_array)
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

	strcpy(path1,path);

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

	/*
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
	*/


	data_array_row_num = map_array[rank];

	*data_array = (double *)malloc(data_array_row_num*sizeof(double));

	long rest_dims = 1;

	for(i=1;i<ndims;i++)
		rest_dims *= dims[i];

	start_1D_id = 0/(rest_dims);
	end_1D_id = (0+data_array_row_num-1)/(rest_dims);

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

	strcpy(path1,path);

	file_id = H5Fopen (path1, H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset_id = H5Dopen (file_id, var_name, H5P_DEFAULT);
	dataspace_id = H5Dget_space(dataset_id);

	memspace_id = H5Screate_simple(ndims,memdims,NULL);

	for(i=0;i<ndims;i++)
		assert(count[i]>0);

	MPI_Barrier(MPI_COMM_WORLD);
	H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,stride,count,block);
	H5Dread(dataset_id,H5T_IEEE_F32LE,memspace_id,dataspace_id,H5P_DEFAULT,buff);
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
		(*data_array)[t_i++] = buff[i];
	free(buff);
}

//read double hdf5
void read_hdf5_double_input_file(char *path,double **data_array)
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

	strcpy(path1,path);

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

	/*
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
	*/


	data_array_row_num = map_array[rank];

	*data_array = (double *)malloc(data_array_row_num*sizeof(double));

	long rest_dims = 1;

	for(i=1;i<ndims;i++)
		rest_dims *= dims[i];

	start_1D_id = 0/(rest_dims);
	end_1D_id = (0+data_array_row_num-1)/(rest_dims);

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

	strcpy(path1,path);

	file_id = H5Fopen (path1, H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset_id = H5Dopen (file_id, var_name, H5P_DEFAULT);
	dataspace_id = H5Dget_space(dataset_id);

	memspace_id = H5Screate_simple(ndims,memdims,NULL);

	for(i=0;i<ndims;i++)
		assert(count[i]>0);

	MPI_Barrier(MPI_COMM_WORLD);
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
		(*data_array)[t_i++] = buff[i];
	}

	free(buff);

}
#endif



#ifdef _ADD_PNETCDF_
//read pnetcdf input file
void read_pnetcdf_input_file(int from)
{
	if(from==0)
	{
		if(float_flag==1)
			load_netcdf_file_all_float(ori_file,&ori_data_array);
		else
			load_netcdf_file_all_double(ori_file,&ori_data_array);
	}
	else
	{
		if(float_flag==1)
			load_netcdf_file_all_float(cmp_file,&cmp_data_array);
		else
			load_netcdf_file_all_double(cmp_file,&cmp_data_array);
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

	/*
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
	*/

	data_array_row_num = map_array[rank];

	*array = (double *)malloc(data_array_row_num*sizeof(double));

	long rest_dims = 1;

	for(i=1;i<var_ndims;i++)
		rest_dims *=dim_sizes[dimids[i]];

	int start_1D_id;
	int end_1D_id;

	start_1D_id = 0/(rest_dims);
	end_1D_id = (0+data_array_row_num-1)/(rest_dims);

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

	buff = (float *) calloc(count_total, sizeof(float));
	ret = ncmpi_get_vara_float_all(ncfile, var_index, start, count, buff);
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
			(*array)[t_i++] = buff[i];

	ret = ncmpi_close(ncfile);
	free(map_array);
	free(buff);

}

//read double pnetcdf
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

	/*
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
	*/

	data_array_row_num = map_array[rank];

	*array = (double *)malloc(data_array_row_num*sizeof(double));

	long rest_dims = 1;

	for(i=1;i<var_ndims;i++)
		rest_dims *=dim_sizes[dimids[i]];

	int start_1D_id;
	int end_1D_id;

	start_1D_id = 0/(rest_dims);
	end_1D_id = (0+data_array_row_num-1)/(rest_dims);

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
			(*array)[t_i++] = buff[i];

	ret = ncmpi_close(ncfile);
	free(map_array);
	free(buff);

}

#endif


#ifdef _ADD_PNETCDF_
void random_de_comp_double()
{

	uLong blen;
	int index_table_sum;
	int uncompress_table_index;
	int uncompress_table_length;
	unsigned char buff[1024*1024];
	unsigned char data_buff[1024*1024];
	unsigned char ch_index[4];
	unsigned int class_id;
	int helper_i,block_p;
	int data_array_index;
	int random_array_index;
	int t_length;
	struct timeval tol_start,tol_end,sub_start,sub_end;
	long sub_total;


	MPI_Offset *dim_sizes, var_size,t_var_size,total_var_size;
	int ncfile, ndims, nvars, ngatts, unlimited;
	int ret;
	char path[128];
	char ch[5];
	int i,j;

	strcpy(path,numarck_file);

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

	B = (int)log((float)dim_sizes[1]);
	block_array_size = dim_sizes[2];
	incompressible_data_num = dim_sizes[4];


	ret = ncmpi_get_att_int(ncfile,0,"elements_per_block",&block_size);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);


	MPI_Offset start[1];
	MPI_Offset count[1];

	offset_block_length = (int *)malloc(block_array_size*sizeof(int));
	offset_incompressible = (int *)malloc(block_array_size*sizeof(int));

	start[0] = 0;
	count[0] = block_array_size;

	ret = ncmpi_get_vara_int_all(ncfile, 2, start, count, offset_block_length);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_get_vara_int_all(ncfile, 3, start, count, offset_incompressible);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);
	
	B = -1;

	for(i=0;i<30;i++)
	{
		if(pow(2,i)==dim_sizes[1])
			B = i;
	}

	class_num = pow(2,B);

	double *t_incompressible_data_array;
	double *t_bin_center;
	double *bin_center;
	double *incompressible_data_array;

	start[0] = 0;
	count[0] = class_num;

	bin_center = (double *)malloc(class_num*sizeof(double));
	t_bin_center = (double *)malloc(class_num*sizeof(double));

	ret = ncmpi_get_vara_double_all(ncfile, 1, start, count, t_bin_center);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	for(i=0;i<class_num;i++)
		bin_center[i] = t_bin_center[i];


	incompressible_data_array = (double *)malloc(incompressible_data_num*sizeof(double));
	t_incompressible_data_array = (double *)malloc(incompressible_data_num*sizeof(double));


	start[0] = 0;
	count[0] = incompressible_data_num;

	ret = ncmpi_get_vara_double_all(ncfile, 5, start, count, t_incompressible_data_array);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	for(i=0;i<incompressible_data_num;i++)
		incompressible_data_array[i] = t_incompressible_data_array[i];

	free(t_incompressible_data_array);

	data_array_index = 0;
	index_table_sum = 0;
	random_array_index = 0;

	helper_array_calc_decomp();

	sub_total = 0;

//	printf("block_array_size = %d\n",block_array_size);
	gettimeofday(&tol_start,NULL);
	int t_id = 0;

	for(i=0;i<block_array_size;i++)
	{
		if(block_overlap(random_start,random_end,i*block_size,(i+1)*block_size))
		{
			start[0] = (MPI_Offset)index_table_sum;
			count[0] = (MPI_Offset)offset_block_length[i];
			count[1] = 0;

			gettimeofday(&sub_start,NULL);
			ret = ncmpi_get_vara_uchar_all(ncfile, 4, start, count, buff);
			if (ret != NC_NOERR) handle_error(ret, __LINE__);
			gettimeofday(&sub_end,NULL);
			sub_total += time_diff(sub_start,sub_end);

//			printf("data_array_row_num = %d\n",data_array_row_num);

			if((i+1)*block_size>data_array_row_num)
				t_length = data_array_row_num - i*block_size;
			else
				t_length = block_size;


			uncompress_table_length = t_length*B/8;

			if((t_length*B)%8!=0)
				uncompress_table_length++;

			blen = uncompress_table_length;

//			printf("block size = %d   uncompress_table _length = %d\n",t_length,uncompress_table_length);

			if(uncompress(data_buff,&blen,buff,offset_block_length[i])!=Z_OK)
			{

				int ret;

				ret = uncompress(data_buff,&blen,buff,offset_block_length[i]);
				switch(ret)
				{
					case Z_BUF_ERROR:
						printf("Z_buff\n");
						break;
					case Z_DATA_ERROR:
						printf("Z_DATA_ERROR\n");
						break;
					case Z_MEM_ERROR:
						printf("Z_MEM_ERROR\n");
						break;
					default:
						printf("?\n");
				}
				printf("zlib uncompress fail!\n");
				exit(1);
			}



			uncompress_table_index = offset_incompressible[i];
			data_array_index = i*block_size;

			helper_i = 0;
			block_p = 0;
			for(j=0;j<t_length;j++)
			{
				ch_index[0] = '\0';
				ch_index[1] = '\0';
				ch_index[3] = '\0';
				ch_index[4] = '\0';

				ch_index[0] |= (data_buff[block_p/8 + 0] & helper_array_mask[helper_i][0])>>helper_array[helper_i][0];
				ch_index[0] |= (data_buff[block_p/8 + 1] & helper_array_mask[helper_i][1])<<helper_array[helper_i][1];
				ch_index[1] |= (data_buff[block_p/8 + 1] & helper_array_mask[helper_i][2])>>helper_array[helper_i][2];
				ch_index[1] |= (data_buff[block_p/8 + 2] & helper_array_mask[helper_i][3])<<helper_array[helper_i][3];

				/*
				if(j==128)
				{
					printf("helper_i = %d\n",helper_i);
					printf("ori data = \n");
					debug_byte(data_buff[block_p/8+0]);
					debug_byte(data_buff[block_p/8+1]);
					debug_byte(data_buff[block_p/8+2]);

					printf("mask\n");
					debug_byte(helper_array_mask[helper_i][0]);
					debug_byte(helper_array_mask[helper_i][1]);
					debug_byte(helper_array_mask[helper_i][2]);
					debug_byte(helper_array_mask[helper_i][3]);

					printf("current = \n");
					debug_byte(ch_index[0]);
					debug_byte(ch_index[1]);
					debug_byte(ch_index[2]);
					debug_byte(ch_index[3]);

					memcpy(&class_id,ch_index,4);
					printf("current id = %d\n");
				}
				*/


				block_p += B;
				helper_i = (block_p) %8;

//				int p_helper_i;

//				p_helper_i = helper_i;


				memcpy(&class_id,ch_index,4);
				assert(class_id<=1024*2*2*2*2);

				if(data_array_index>=random_start && data_array_index<=random_end)
				{
					if(class_id==class_num-1)
					{
						data_array[random_array_index] = incompressible_data_array[uncompress_table_index];
					}
					else
					{
						data_array[random_array_index] = ori_data_array[data_array_index]*(1.0+bin_center[class_id]);
					}

					random_array_index++;
					index_id[t_id++] = class_id;
				}

				if(class_id==class_num-1)
					uncompress_table_index ++;
				data_array_index++;
			}

		}

		index_table_sum += offset_block_length[i];
	}
	gettimeofday(&tol_end,NULL);
//	printf("partial decomp %ld\n",time_diff(tol_start,tol_end)-sub_total);

	ret = ncmpi_close(ncfile);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

}

void random_de_comp_float()
{

	uLong blen;
	int index_table_sum;
	int uncompress_table_index;
	int uncompress_table_length;
	unsigned char buff[1024*1024];
	unsigned char data_buff[1024*1024];
	unsigned char ch_index[4];
	unsigned int class_id;
	int helper_i,block_p;
	int data_array_index;
	int random_array_index;
	int t_length;
	struct timeval tol_start,tol_end,sub_start,sub_end;
	long sub_total;

	MPI_Offset *dim_sizes, var_size,t_var_size,total_var_size;
	int ncfile, ndims, nvars, ngatts, unlimited;
	int ret;
	char path[128];
	char ch[5];
	int i,j;


	strcpy(path,numarck_file);

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

	B = (int)log((float)dim_sizes[1]);
	block_array_size = dim_sizes[2];
	incompressible_data_num = dim_sizes[4];


	ret = ncmpi_get_att_int(ncfile,0,"elements_per_block",&block_size);
	if(ret != NC_NOERR) handle_error(ret, __LINE__);


	MPI_Offset start[1];
	MPI_Offset count[1];

	offset_block_length = (int *)malloc(block_array_size*sizeof(int));
	offset_incompressible = (int *)malloc(block_array_size*sizeof(int));

	start[0] = 0;
	count[0] = block_array_size;

	ret = ncmpi_get_vara_int_all(ncfile, 2, start, count, offset_block_length);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	ret = ncmpi_get_vara_int_all(ncfile, 3, start, count, offset_incompressible);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);
	
	B = -1;

	for(i=0;i<30;i++)
	{
		if(pow(2,i)==dim_sizes[1])
			B = i;
	}

	class_num = pow(2,B);

	float *t_incompressible_data_array;
	float *t_bin_center;
	float *bin_center;
	float *incompressible_data_array;

	start[0] = 0;
	count[0] = class_num;

	bin_center = (float *)malloc(class_num*sizeof(float));
	t_bin_center = (float *)malloc(class_num*sizeof(float));

	ret = ncmpi_get_vara_float_all(ncfile, 1, start, count, t_bin_center);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	for(i=0;i<class_num;i++)
		bin_center[i] = t_bin_center[i];

	incompressible_data_array = (float *)malloc(incompressible_data_num*sizeof(float));
	t_incompressible_data_array = (float *)malloc(incompressible_data_num*sizeof(float));


	start[0] = 0;
	count[0] = incompressible_data_num;

	ret = ncmpi_get_vara_float_all(ncfile, 5, start, count, t_incompressible_data_array);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

	for(i=0;i<incompressible_data_num;i++)
		incompressible_data_array[i] = t_incompressible_data_array[i];

	free(t_incompressible_data_array);

	data_array_index = 0;
	index_table_sum = 0;
	random_array_index = 0;

	helper_array_calc_decomp();

	sub_total = 0;

//	printf("block_array_size = %d\n",block_array_size);
	gettimeofday(&tol_start,NULL);
	int t_id = 0;

	for(i=0;i<block_array_size;i++)
	{
		if(block_overlap(random_start,random_end,i*block_size,(i+1)*block_size))
		{
			start[0] = (MPI_Offset)index_table_sum;
			start[1] = 0;
			count[0] = (MPI_Offset)offset_block_length[i];
			count[1] = 0;

			gettimeofday(&sub_start,NULL);
			ret = ncmpi_get_vara_schar_all(ncfile, 4, start, count, buff);
			if (ret != NC_NOERR) handle_error(ret, __LINE__);

			gettimeofday(&sub_end,NULL);
			sub_total += time_diff(sub_start,sub_end);

//			printf("data_array_row_num = %d\n",data_array_row_num);

			if((i+1)*block_size>data_array_row_num)
				t_length = data_array_row_num - i*block_size;
			else
				t_length = block_size;

			uncompress_table_length = t_length*B/8;

			if((t_length*B)%8!=0)
				uncompress_table_length++;

			blen = uncompress_table_length;
			

			z_stream strm;
			strm.zalloc = Z_NULL;
			strm.zfree = Z_NULL;
			strm.opaque = Z_NULL;
			strm.next_in = buff;
			ret = inflateInit(&strm);
			assert(ret==Z_OK);
			strm.avail_in = offset_block_length[i];
			strm.avail_out = blen;
			strm.next_out = data_buff;
			ret = inflate(&strm, Z_NO_FLUSH);
			assert(ret != Z_STREAM_ERROR);

			/*
			if(uncompress(data_buff,&blen,buff,offset_block_length[i])!=Z_OK)
			{

				int ret;

				printf("i = %d block length = %d\n",i,offset_block_length[i]);
				ret = uncompress(data_buff,&blen,buff,offset_block_length[i]);
				switch(ret)
				{
					case Z_BUF_ERROR:
						printf("Z_buff\n");
						break;
					case Z_DATA_ERROR:
						printf("Z_DATA_ERROR\n");
						break;
					case Z_MEM_ERROR:
						printf("Z_MEM_ERROR\n");
						break;
					default:
						printf("?\n");
				}
				printf("zlib uncompress fail!\n");
				exit(1);
			}
			*/



			uncompress_table_index = offset_incompressible[i];
			data_array_index = i*block_size;

			helper_i = 0;
			block_p = 0;
			for(j=0;j<t_length;j++)
			{
				ch_index[0] = '\0';
				ch_index[1] = '\0';
				ch_index[2] = '\0';
				ch_index[3] = '\0';

				ch_index[0] |= (data_buff[block_p/8 + 0] & helper_array_mask[helper_i][0])>>helper_array[helper_i][0];
				ch_index[0] |= (data_buff[block_p/8 + 1] & helper_array_mask[helper_i][1])<<helper_array[helper_i][1];
				ch_index[1] |= (data_buff[block_p/8 + 1] & helper_array_mask[helper_i][2])>>helper_array[helper_i][2];
				ch_index[1] |= (data_buff[block_p/8 + 2] & helper_array_mask[helper_i][3])<<helper_array[helper_i][3];

				/*
				if(j==128)
				{
					printf("helper_i = %d\n",helper_i);
					printf("ori data = \n");
					debug_byte(data_buff[block_p/8+0]);
					debug_byte(data_buff[block_p/8+1]);
					debug_byte(data_buff[block_p/8+2]);

					printf("mask\n");
					debug_byte(helper_array_mask[helper_i][0]);
					debug_byte(helper_array_mask[helper_i][1]);
					debug_byte(helper_array_mask[helper_i][2]);
					debug_byte(helper_array_mask[helper_i][3]);

					printf("current = \n");
					debug_byte(ch_index[0]);
					debug_byte(ch_index[1]);
					debug_byte(ch_index[2]);
					debug_byte(ch_index[3]);

					memcpy(&class_id,ch_index,4);
					printf("current id = %d\n");
				}
				*/


				block_p += B;
				helper_i = (block_p) %8;

//				int p_helper_i;

//				p_helper_i = helper_i;


				memcpy(&class_id,ch_index,4);

				if(data_array_index>=random_start && data_array_index<=random_end)
				{
					if(class_id==class_num-1)
					{
						data_array[random_array_index] = incompressible_data_array[uncompress_table_index];
					}
					else
					{
						data_array[random_array_index] = ori_data_array[data_array_index]*(1.0+bin_center[class_id]);
					}

					random_array_index++;
					index_id[t_id++] = class_id;
				}

				if(class_id==class_num-1)
					uncompress_table_index ++;
				data_array_index++;
			}

		}

		index_table_sum += offset_block_length[i];
	}
	gettimeofday(&tol_end,NULL);
	printf("decompression time = %ld\n",time_diff(tol_start,tol_end)-sub_total);

	ret = ncmpi_close(ncfile);
	if (ret != NC_NOERR) handle_error(ret, __LINE__);

}
#endif

int block_overlap(int a,int b,int c,int d)
{
	if(a<=c&&b>=d)
		return 1;
	if(c<=a&&d>=b)
		return 1;
	if(c<=b&&d>=a)
		return 1;
	if(a<=d&&b>=c)
		return 1;
	return 0;
}

void helper_array_calc_decomp()
{
	int r_length;
	int i;


	for(i=0;i<8;i++)
	{
		r_length = B;

//		helper_array[i][0] = int_max(i,8-r_length);
		helper_array[i][0] = i;
		helper_array_mask[i][0] = mask_gen(r_length,helper_array[i][0],0);
		r_length -= (8-helper_array[i][0]);
//		debug_byte(helper_array_mask[i][0]);
//		printf("r_length = %d,helper_array %d\n",r_length,helper_array[i][0]);
		if(r_length<0)
			r_length = 0;
//		printf("%d\n",helper_array[i][0]);


//		helper_array[i][1] = int_max(8-i,8-r_length);
		helper_array[i][1] = 8-i;
		helper_array_mask[i][1] = mask_gen(r_length,helper_array[i][1]%8,1);
		r_length -= (8-helper_array[i][1]);
//		debug_byte(helper_array_mask[i][1]);
//		printf("r_length = %d,helper_array %d\n",r_length,helper_array[i][1]);
		if(r_length<0)
			r_length = 0;
//		printf("%d\n",helper_array[i][1]);

//		helper_array[i][2] = int_max(i,8-r_length);
		helper_array[i][2] = i;
		helper_array_mask[i][2] = mask_gen(r_length,helper_array[i][2],2);
		r_length -= (8-helper_array[i][2]);
//		debug_byte(helper_array_mask[i][2]);
//		printf("r_length = %d,helper_array %d\n",r_length,helper_array[i][2]);
		if(r_length<0)
			r_length = 0;
//		printf("%d\n",helper_array[i][2]);

//		helper_array[i][3] = int_max(8-i,8-r_length);
		helper_array[i][3] = 8-i;
		helper_array_mask[i][3] = mask_gen(r_length,helper_array[i][3]%8,3);
		r_length -= (8-helper_array[i][3]);
//		debug_byte(helper_array_mask[i][3]);
//		printf("r_length = %d,helper_array %d\n",r_length,helper_array[i][3]);
		if(r_length<0)
			r_length = 0;
//		printf("%d\n",helper_array[i][3]);

	}



//	helper_array_mask[1][0] = 0xFE;
//	helper_array_mask[1][1] = 0x01;
//	helper_array_mask[1][2] = 0xF8;
//	helper_array_mask[1][3] = 0x00;


//	helper_array_mask[2][2] = 0xFF;
//	helper_array_mask[2][3] = 0x00;


//	helper_array_mask[4][2] = 0xFF;
//	helper_array_mask[4][3] = 0x03;



//	helper_array_mask[6][2] = 0xFF;
//	helper_array_mask[6][3] = 0x0F;

}

/*
void helper_array_calc_decomp()
{
	int r_length;
	int i;


	for(i=0;i<8;i++)
	{
		r_length = B;

//		helper_array[i][0] = int_max(i,8-r_length);
		helper_array[i][0] = i;
		helper_array_mask[i][0] = mask_gen(r_length,helper_array[i][0],0);
		r_length -= (8-helper_array[i][0]);
//		debug_byte(helper_array_mask[i][0]);
//		printf("r_length = %d,helper_array %d\n",r_length,helper_array[i][0]);
		if(r_length<0)
			r_length = 0;
//		printf("%d\n",helper_array[i][0]);


//		helper_array[i][1] = int_max(8-i,8-r_length);
		helper_array[i][1] = 8-i;
		helper_array_mask[i][1] = mask_gen(r_length,helper_array[i][1],1);
		r_length -= (8-helper_array[i][1]);
//		debug_byte(helper_array_mask[i][1]);
//		printf("r_length = %d,helper_array %d\n",r_length,helper_array[i][1]);
		if(r_length<0)
			r_length = 0;
//		printf("%d\n",helper_array[i][1]);

//		helper_array[i][2] = int_max(i,8-r_length);
		helper_array[i][2] = i;
		helper_array_mask[i][2] = mask_gen(r_length,helper_array[i][2],2);
		r_length -= (8-helper_array[i][2]);
//		debug_byte(helper_array_mask[i][2]);
//		printf("r_length = %d,helper_array %d\n",r_length,helper_array[i][2]);
		if(r_length<0)
			r_length = 0;
//		printf("%d\n",helper_array[i][2]);

//		helper_array[i][3] = int_max(8-i,8-r_length);
		helper_array[i][3] = 8-i;
		helper_array_mask[i][3] = mask_gen(r_length,helper_array[i][3],3);
		r_length -= (8-helper_array[i][3]);
//		debug_byte(helper_array_mask[i][3]);
//		printf("r_length = %d,helper_array %d\n",r_length,helper_array[i][3]);
		if(r_length<0)
			r_length = 0;
//		printf("%d\n",helper_array[i][3]);

	}



//	helper_array_mask[1][0] = 0xFE;
//	helper_array_mask[1][1] = 0x01;
//	helper_array_mask[1][2] = 0xF8;
//	helper_array_mask[1][3] = 0x00;


//	helper_array_mask[2][2] = 0xFF;
//	helper_array_mask[2][3] = 0x00;


//	helper_array_mask[4][2] = 0xFF;
//	helper_array_mask[4][3] = 0x03;



//	helper_array_mask[6][2] = 0xFF;
//	helper_array_mask[6][3] = 0x0F;

}
*/

unsigned char mask_gen(int r_length,int offset,int flag)
{
	unsigned char t = 0x01;
	unsigned char ret = 0x00;
	int t_length;
	int i;

	flag %=2;

	if(r_length<0)
		r_length = 0;

	t_length = r_length<(8-offset)?r_length:(8-offset);

	/*
	if(offset==0)
	{
		for(i=0;i<t_length;i++)
			ret |= t<<i;
		return ret;
	}
	*/

	if(flag==0)
	{
		for(i=0;i<t_length;i++)
			ret |= t<<(7-i);
		if(r_length<8-offset)
			ret >>= (8-offset) - r_length;
	}
	else
	{
		for(i=0;i<t_length;i++)
			ret |= t<<i;
	}

	return ret;
}

/*
unsigned char mask_gen(int r_length,int offset,int flag)
{
	unsigned char t = 0x01;
	unsigned char ret = 0x00;
	int t_length;
	int i;

	flag %=2;

	if(r_length<0)
		r_length = 0;

	t_length = r_length<(8-offset)?r_length:(8-offset);

	if(offset==0)
	{
		for(i=0;i<t_length;i++)
			ret |= t<<i;
		return ret;
	}

	if(flag==0)
	{
		for(i=0;i<t_length;i++)
			ret |= t<<(7-i);
	}
	else
	{
		for(i=0;i<t_length;i++)
			ret |= t<<i;
	}

	return ret;
}
*/

static void handle_error(int status, int lineno)
{
	fprintf(stderr, "Error at line %d: %s\n", lineno, ncmpi_strerror(status));
	MPI_Abort(MPI_COMM_WORLD, 1);
}

long int time_diff(struct timeval t_s,struct timeval t_e)
{
	long int ret;

	ret = t_e.tv_sec - t_s.tv_sec;

	ret = ret*1000000;
	ret = ret + (t_e.tv_usec - t_s.tv_usec);

	if(ret<0)
	{
		printf("???");
		exit(0);
	}
//	return ret/1000;
	return ret;
}

float diff(float a,float b)
{
	if(a>=b)
		return a-b;
	else
		return b-a;
}

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

void cmd_helper()
{
	char *help =
		"Usage: mpirun -1 ./exe_mpi_decompress -i 'ori_file_name' -j 'cmp_file_name' -o 'NUMARCK_file_name' -v 'varable name'\n"
		"-i ori_file_name	: first iteration file name\n"
		"-j cmp_file_name	: second iteration file name (comparison purpose)\n"
		"-o NUMARCK_file_name	: NUMARCK file name\n"
		"-v varable name		: varable name in input file\n"
		"-d			: elements are double data type (default is float)\n"
		"-x			: input file format is hdf5 (default is Pnetcdf)\n"
		"-y			: numarck file format is hdf5 (default is Pnetcdf, not supported in this version!)\n"
		"-s start_index		: start index of decompressed data (partial decompression, default is 0)\n"
		"-e end_index		: end index of decompressed data (partial decompression, default is total_data_element_num)\n"
		"-h			: print this help information\n";

	if(rank==0)
		printf("%s\n",help);

}

void itoa(char *ch,int num)
{
	ch[0] = '0' + (num/100)%10;
	ch[1] = '0' + (num/10)%10;
	ch[2] = '0' + num%10;
	ch[3] = '\0';
}
