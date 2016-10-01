#ifndef HEAP_H_INCLUDED
#define HEAP_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE 10000000

struct t_center_element
{
	double center;
	double sum;
	int count;
	double g_sum;
	int g_count;
	int index;
	int validate_flag;
	int selected_flag;
	int class_id;
};

struct t_center_element heap[MAX_SIZE+1];

//int heap[MAX_SIZE];
void min_Heap_insert(struct t_center_element *heap,int *n,struct t_center_element item);
struct t_center_element min_Heap_delete(struct t_center_element *heap,int *n);
struct t_center_element min_Heap_get(struct t_center_element *heap,int *n);
#endif // HEAP_H_INCLUDED


   /*最大堆的插入操作*/
   /*注：堆的下标是从1开始,而不是0*/
	/*changed into min heap*/
   void min_Heap_insert(struct t_center_element *heap,int *n,struct t_center_element item)
{
	    int i,parent;//i为当前节点,parent为i的父节点
	        if((*n)>=MAX_SIZE)//堆为满
			    {
				            printf("The heap is full\n");
					            exit(1);
						        }
		    i=++(*n);
		        parent=i/2;
			    while((i!=1) && (item.g_count<heap[parent].g_count))//若堆为非空,且所插入数据item大于父节点的关键字值
				        {
						        heap[i]=heap[parent];//父节点关键字值下移
							        i=parent;//把父节点作为当前节点
								        parent/=2;//依次求父节点
									    }
			        heap[i]=item;//插入到正确的位置
}
/*最大堆的删除操作*/
struct t_center_element min_Heap_delete(struct t_center_element *heap,int *n)
{
	    struct t_center_element item,temp;
	        int child,parent;
		    if(*n==0)//若为空堆
			        {
					        printf("The heap is empty.\n");
						        exit(1);
							    }
		        item=heap[1];//把最大堆的最大元素赋给item
			    temp=heap[(*n)--];//堆的最后节点关键字值
			        parent=1;
				    child=2*parent;
				        while(child<=(*n))
						    {
							            if(child<*n && heap[child].g_count>heap[child+1].g_count)
									                child++;//找出堆中最大关键字值的节点
								            if(temp.g_count<=heap[child].g_count)break;//把最大节点关键字值与最后节点关键字值比较
									            else
											            {//若堆中存在比最后节点关键字值大的节点,则交换位置
													                heap[parent]=heap[child];
															            parent=child;
																                child*=2;
																		        }
										        }
					    heap[parent]=temp;//插入到正确位置
					        return item;//返回删除的关键字值
}

struct t_center_element min_Heap_get(struct t_center_element *heap,int *n)
{
	    struct t_center_element item,temp;
		    if(*n==0)//若为空堆
			        {
					        printf("The heap is empty.\n");
						        exit(1);
							    }
		        item=heap[1];//把最大堆的最大元素赋给item
					        return item;//返回删除的关键字值
}


/*
int main()
{
	    int item,i;
	        int n=0;
		    for(i=1;i<MAX_SIZE;i++)
			        {
					        scanf("%d",&item);
						        max_Heap_insert(heap,&n,item);
							    }
		        for(i=1;i<=n;i++)
				        printf("%d ",heap[i]);
			    printf("\n");
			        item=max_Heap_delete(heap,&n);
				    printf("The deleted data is:%d",item);
				        printf("\n");
					    for(i=1;i<=n;i++)
						            printf("%d ",heap[i]);
					        return 0;
}
*/
