#include "stdlib.h"
#include "stddef.h"
#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "limits.h"


#define PREPARE_FIRST_COPY()                                      \
    do {                                                          \
    if (src_len >= (CHAR_BIT - dst_offset_modulo)) {              \
        *dst     &= reverse_mask[dst_offset_modulo];              \
        src_len -= CHAR_BIT - dst_offset_modulo;                  \
    } else {                                                      \
        *dst     &= reverse_mask[dst_offset_modulo]               \
              | reverse_mask_xor[dst_offset_modulo + src_len];    \
         c       &= reverse_mask[dst_offset_modulo + src_len];    \
        src_len = 0;                                              \
    } } while (0)


int write_unit_length;
int write_buff_unit_num;
long write_buff_length;
unsigned char *write_bit_buff;

int read_unit_length;
int read_buff_unit_num;
unsigned char *read_bit_buff;

long write_bit_buff_init(int unit_length,int unit_num);
void write_unit(int value,int index);
void write_one_bit(int value,long bit_pos);
unsigned char *get_write_bit_buff();
void read_bit_buff_init(unsigned char *buff, int unit_length, int buff_unit_num);
int read_unit(int index);
int read_one_bit(int bit_pos);
void write_buff_debug();
void bits_copy(unsigned char *src_org, unsigned char *dst_org,long src_offset, long dst_offset,long src_len);
void bits_seg_copy(unsigned char *src,unsigned char *dest,long src_p,long dest_p,int l);
void bit_copy(unsigned char *src,unsigned char *dest,long src_p,long dest_p);
int get_write_bit_buff_bit_length();
void bits_tail_copy(unsigned char *src,unsigned char *dest,long src_p,long dest_p,long l);
void buff_debug(unsigned char *buff);

/*
int main()
{
	unsigned char *buff;
	int i;

	write_bit_buff_init(5,50);

	for(i=0;i<50;i++)
		write_unit(i+1,i);

//	write_buff_debug();

	buff = get_write_bit_buff();

	read_bit_buff_init(buff,5,50);

	for(i=0;i<50;i++)
		printf("%d\n",read_unit(i));

	return 0;
}
*/

long write_bit_buff_init(int unit_length,int unit_num)
{
	write_unit_length = unit_length;
	write_buff_unit_num = unit_num;

	write_buff_length = (long)unit_length*unit_num/8;
	if((unit_length*unit_num)%8!=0)
		write_buff_length++;

	write_bit_buff = (unsigned char *)calloc(write_buff_length,sizeof(unsigned char));
	return write_buff_length;
}

void write_unit(int value,int index)
{
	unsigned char buff_value[sizeof(int)];
	unsigned char flag;
	long i;

	if(index>=write_buff_unit_num)
	{
		printf("write read index = %d , max = %d\n",index,write_buff_unit_num);
		exit(1);
	}

	memcpy(buff_value,&value,sizeof(int));

	bits_copy(buff_value,write_bit_buff,0,index*write_unit_length,write_unit_length);

	/*
	for(i=0;i<write_unit_length;i++)
	{
		flag = 0x01 << (i%8);
		if(buff_value[i/8]&flag)
			write_one_bit(1,(long)index*(long)write_unit_length+i);
		else
			write_one_bit(0,(long)index*(long)write_unit_length+i);
	}
	*/
}

void write_one_bit(int value,long bit_pos)
{
	unsigned char flag;
	long byte;
	int bit;

	byte = bit_pos/8;
	bit = bit_pos%8;

	flag = 0x01;
	flag = flag << bit;

	if(value==0)
		write_bit_buff[byte] &= (~flag);
	else
		write_bit_buff[byte] |= (flag);
}

unsigned char *get_write_bit_buff()
{
	return write_bit_buff;
}

int get_write_bit_buff_bit_length()
{
	return write_unit_length*write_buff_unit_num;
}





/*
void bits_copy(unsigned char *src,unsigned char *dest,long src_p,long dest_p,long l)
{
	int phase_1_bits,phase_2_bits;
	long count;
	long i;

	if(l==0)
		return;

	count = l;

	int t;

	t = 8-(src_p%8);

	if(t!=8)
	{
		for(i=0;i<t;i++)
		{
			bit_copy(src,dest,src_p,dest_p);
			src_p ++;
			dest_p ++;
			count --;
			if(count==0)
				break;
		}
	}

	while(count>=8)
	{
		phase_1_bits = 8 - (dest_p%8);
		if(phase_1_bits)
			bits_seg_copy(src,dest,src_p,dest_p,phase_1_bits);
		src_p += phase_1_bits;
		dest_p += phase_1_bits;

		phase_2_bits = 8 - phase_1_bits;
		if(phase_2_bits)
			bits_seg_copy(src,dest,src_p,dest_p,phase_2_bits);
		src_p += phase_2_bits;
		dest_p += phase_2_bits;

		count -= 8;
	}

	if(count>0)
		bits_tail_copy(src,dest,src_p,dest_p,count);

}
*/

void bits_tail_copy(unsigned char *src,unsigned char *dest,long src_p,long dest_p,long l)
{
	unsigned char flag;
	unsigned char dest_flag;
	unsigned char temp;

	assert((src_p%8==0) && (l < 8));

	switch(l)
	{
		case 1:
			flag = 0x01;
			break;
		case 2:
			flag = 0x03;
			break;
		case 3:
			flag = 0x07;
			break;
		case 4:
			flag = 0x0F;
			break;
		case 5:
			flag = 0x1F;
			break;
		case 6:
			flag = 0x3F;
			break;
		case 7:
			flag = 0x7F;
			break;
		case 8:
			flag = 0xFF;
			break;
		default:
			printf("bits_seg_copy error!\n");
			exit(1);
	}

	temp = src[src_p/8] & flag;
	temp = temp << (dest_p%8);

	dest_flag = flag<<(dest_p%8);
	dest[dest_p/8] &= ~(dest_flag);
	dest[dest_p/8] |= temp;

}

void bits_copy(unsigned char *src_org, unsigned char *dst_org,long src_offset, long dst_offset,long src_len)
{
    static const unsigned char mask[] =
        { 0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff };
    static const unsigned char reverse_mask[] =
        { 0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, 0xff };
    static const unsigned char reverse_mask_xor[] =
        { 0xff, 0x7f, 0x3f, 0x1f, 0x0f, 0x07, 0x03, 0x01, 0x00 };

    if (src_len) {
        const unsigned char *src;
              unsigned char *dst;
        int                  src_offset_modulo,
                             dst_offset_modulo;

        src = src_org + (src_offset / CHAR_BIT);
        dst = dst_org + (dst_offset / CHAR_BIT);

        src_offset_modulo = src_offset % CHAR_BIT;
        dst_offset_modulo = dst_offset % CHAR_BIT;

        if (src_offset_modulo == dst_offset_modulo) {
            long              byte_len;
            long              src_len_modulo;
            if (src_offset_modulo) {
                unsigned char   c;

                c = reverse_mask_xor[dst_offset_modulo]     & *src++;

                PREPARE_FIRST_COPY();
                *dst++ |= c;
            }

            byte_len = src_len / CHAR_BIT;
            src_len_modulo = src_len % CHAR_BIT;

            if (byte_len) {
                memcpy(dst, src, byte_len);
                src += byte_len;
                dst += byte_len;
            }
            if (src_len_modulo) {
                *dst     &= reverse_mask_xor[src_len_modulo];
                *dst |= reverse_mask[src_len_modulo]     & *src;
            }
        } else {
            long             bit_diff_ls,
                            bit_diff_rs;
            long             byte_len;
            int             src_len_modulo;
            unsigned char   c;
            /*
             * Begin: Line things up on destination. 
             */
            if (src_offset_modulo > dst_offset_modulo) {
                bit_diff_ls = src_offset_modulo - dst_offset_modulo;
                bit_diff_rs = CHAR_BIT - bit_diff_ls;

                c = *src++ << bit_diff_ls;
                c |= *src >> bit_diff_rs;
                c     &= reverse_mask_xor[dst_offset_modulo];
            } else {
                bit_diff_rs = dst_offset_modulo - src_offset_modulo;
                bit_diff_ls = CHAR_BIT - bit_diff_rs;

                c = *src >> bit_diff_rs     &
                    reverse_mask_xor[dst_offset_modulo];
            }
            PREPARE_FIRST_COPY();
            *dst++ |= c;

            /*
             * Middle: copy with only shifting the source. 
             */
            byte_len = src_len / CHAR_BIT;

            while (--byte_len >= 0) {
                c = *src++ << bit_diff_ls;
                c |= *src >> bit_diff_rs;
                *dst++ = c;
            }

            /*
             * End: copy the remaing bits; 
             */
            src_len_modulo = src_len % CHAR_BIT;
            if (src_len_modulo) {
                c = *src++ << bit_diff_ls;
                c |= *src >> bit_diff_rs;
                c     &= reverse_mask[src_len_modulo];

                *dst     &= reverse_mask_xor[src_len_modulo];
                *dst |= c;
            }
        }
    }
}

/*
void bits_copy(unsigned char *src,unsigned char *dest,long src_p,long dest_p,long l)
{
	long i;

	for(i=0;i<l;i++)
		bit_copy(src,dest,src_p+i,dest_p+i);
}
*/

void bits_seg_copy(unsigned char *src,unsigned char *dest,long src_p,long dest_p,int l)
{
	unsigned char flag;
	unsigned char src_flag,dest_flag;
	unsigned char temp;
	long i;

	if(l==0)
		return;
//	printf("src_p = %ld  dest_p = %ld  l = %d\n",src_p,dest_p,l);
	assert(((src_p%8+l)==8) || ((dest_p%8+l)==8));

	switch(l)
	{
		case 1:
			flag = 0x01;
			break;
		case 2:
			flag = 0x03;
			break;
		case 3:
			flag = 0x07;
			break;
		case 4:
			flag = 0x0F;
			break;
		case 5:
			flag = 0x1F;
			break;
		case 6:
			flag = 0x3F;
			break;
		case 7:
			flag = 0x7F;
			break;
		case 8:
			flag = 0xFF;
			break;
		default:
			printf("bits_seg_copy error!\n");
			exit(1);
	}

	if(((src_p%8+l)%8)==0)
	{
		assert((src_p%8)>=(dest_p%8));
		src_flag = flag<<(8 - l);
		temp = src[src_p/8] & src_flag;
		temp = temp << ((src_p%8) - (dest_p%8));

		dest_flag = flag<<(dest_p%8);
		dest[dest_p/8] &= ~(dest_flag);
		dest[dest_p/8] |= temp;
	}
	else
	{
		if((src_p%8)>(dest_p%8))
		assert((src_p%8)<=(dest_p%8));

		dest_flag = flag<<(8 - l);
		dest[dest_p/8] &= ~(dest_flag);

		src_flag = flag<<(src_p%8);

		///////
		temp = src[src_p/8] & src_flag;
		temp = temp << ((dest_p%8) - (src_p%8));
/////
		dest[dest_p/8] |= temp;

	}
}

void bit_copy(unsigned char *src,unsigned char *dest,long src_p,long dest_p)
{
	unsigned char test_flag,write_flag;
	long src_byte,dest_byte;
	long src_bit,dest_bit;

	src_byte = src_p/8;
	src_bit = src_p%8;

	dest_byte = dest_p/8;
	dest_bit = dest_p%8;

	test_flag = 0x01;
	test_flag = test_flag << src_bit;

	write_flag = 0x01;
	write_flag = write_flag << dest_bit;

	if((src[src_byte] & test_flag)==0)
	{
		dest[dest_byte] &= (~write_flag);
	}
	else
	{
		dest[dest_byte] |= (write_flag);
	}

}

void cat_bit(unsigned char *seg,unsigned char *body,int seg_l,int body_l,int flag,unsigned char **ret)
{
	int ret_l,ret_byte_l;

	ret_l = seg_l + body_l;

	ret_byte_l = ret_l/8;
	if(ret_l%8)
		ret_byte_l++;

	*ret = (unsigned char *)malloc(ret_byte_l*sizeof(unsigned char));

	if(flag==0)
	{
		bits_copy(seg,*ret,0,0,seg_l);
		bits_copy(body,*ret,0,seg_l,body_l);
	}
	else
	{
		bits_copy(body,*ret,0,0,body_l);
		bits_copy(seg,*ret,0,body_l,seg_l);
	}
}

void split_bit(unsigned char *input,long input_l,long split_l,int flag,unsigned char **seg,unsigned char **body)
{
	long seg_l,body_l;
	long seg_byte_l,body_byte_l;

	seg_l = split_l;
	body_l = input_l - split_l;

	seg_byte_l = seg_l/8;
	if(seg_l%8)
		seg_byte_l++;

	body_byte_l = body_l/8;
	if(body_l%8)
		body_byte_l++;

	*seg = (unsigned char *)malloc(seg_byte_l*sizeof(unsigned char));
	*body = (unsigned char *)malloc(body_byte_l*sizeof(unsigned char));
	if(flag==0)
	{
		bits_copy(input,*seg,0,0,seg_l);
		bits_copy(input,*body,seg_l,0,body_l);
	}
	else
	{
		bits_copy(input,*body,0,0,body_l);
		bits_copy(input,*seg,body_l,0,seg_l);
	}
}

void read_bit_buff_init(unsigned char *buff, int unit_length, int buff_unit_num)
{
	read_bit_buff = buff;
	read_unit_length = unit_length;
	read_buff_unit_num = buff_unit_num;
}

int read_unit(int index)
{
	unsigned char value_buff[sizeof(int)];
	unsigned char flag;
	int value;
	long i;

	if(index>=read_buff_unit_num)
	{
		printf("illegal read!");
		exit(1);
	}

	memset(value_buff,0,sizeof(int));

	for(i=0;i<read_unit_length;i++)
	{
		flag = 0x01<<(i%8);
		if(read_one_bit(index*read_unit_length+i))
			value_buff[i/8] |= (flag);
	}

	memcpy(&value,value_buff,sizeof(int));

	return value;
}

int read_one_bit(int bit_pos)
{
	unsigned char flag;
	int byte;
	int bit;

	byte = bit_pos/8;
	bit = bit_pos%8;

	flag = 0x01<<bit;

	if(read_bit_buff[byte]&flag)
		return 1;
	else
		return 0;
}

void write_buff_debug()
{
	long i,j;

	for(i=0;i<write_buff_length;i++)
	{
		for(j=0;j<8;j++)
		{
			if(write_bit_buff[i]&(0x01<<j))
				printf("1,");
			else
				printf("0,");
		}
		printf("\n");
	}
}

void buff_debug(unsigned char *buff)
{
	long i,j;

	for(i=0;i<write_buff_length;i++)
	{
		for(j=0;j<8;j++)
		{
			if(buff[i]&(0x01<<j))
				printf("1,");
			else
				printf("0,");
		}
		printf("\n");
	}
}
