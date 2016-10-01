extern long write_bit_buff_init(int unit_length,int unit_num);
extern void write_unit(int value,int index);
extern unsigned char *get_write_bit_buff();
extern void read_bit_buff_init(unsigned char *buff, int unit_length, int buff_unit_num);
extern int read_unit(int index);
extern void cat_bit(unsigned char *seg,unsigned char *body,int seg_l,int body_l,int flag,unsigned char **ret);
extern void split_bit(unsigned char *input,int input_l,int split_l,int flag,unsigned char **seg,unsigned char **body);
extern void bits_copy(unsigned char *src,unsigned char *dest,int src_p,int dest_p,int l);
