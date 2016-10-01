MPICC = mpicc

DEFS = -D_ADD_HDF5_ -D_ADD_PNETCDF_
OPTS = -O2

CFLAGS = $(OPTS) $(DEFS)

PNETCDF_INCFLAGS = 
PNETCDF_LIBS = -lpnetcdf

HDF5_INCFLAGS = 
HDF5_LIBS = -lhdf5_hl -lhdf5

COMM_INCFLAGS = 
COMM_LIBS = -lz -lm

LIBS = $(PNETCDF_INCFLAGS) $(HDF5_INCFLAGS) $(COMM_FLAGS) $(PNETCDF_LIBS) $(HDF5_LIBS) $(COMM_LIBS)

.c.o:
	$(MPICC) $(CFLAGS) -c $<

comp_pnetcdf_hdf5 : mpi_compress.o mpi_kmeans.o memory_bit_buffer.o
	$(MPICC) -D_ADD_HDF5_ -D_ADD_PNETCDF_ -o $@ $^ $(LIBS)

comp_pnetcdf : mpi_compress.o mpi_kmeans.o memory_bit_buffer.o
	$(MPICC) -D_ADD_PNETCDF_ -o $@ $^ $(LIBS)

comp_hdf5 : mpi_compress.o mpi_kmeans.o memory_bit_buffer.o
	$(MPICC) -D_ADD_HDF5_ -o $@ $^ $(LIBS)

decomp_pnetcdf_hdf5 : mpi_decompress.o memory_bit_buffer.o
	$(MPICC) -D_ADD_PNETCDF_ -D_ADD_HDF5_ -o $@ $^ $(LIBS)

decomp_hdf5 : mpi_decompress.o memory_bit_buffer.o
	$(MPICC) -D_ADD_HDF5_ -o $@ $^ $(LIBS)

decomp_pnetcdf : mpi_decompress.c memory_bit_buffer.c
	$(MPICC) -D_ADD_PNETCDF_ -o $@ $^ $(LIBS)

clean:
	rm -f *.o
