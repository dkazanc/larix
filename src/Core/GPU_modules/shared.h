#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8

#define BLKXSIZE2D 16
#define BLKYSIZE2D 16

#define TILEDIMX 32
#define TILEDIMY 16

const int CONSTVECSIZE_9 = 9;
const int CONSTVECSIZE_25 = 25;
const int CONSTVECSIZE_49 = 49;
const int CONSTVECSIZE_81 = 81;
const int CONSTVECSIZE_121 = 121;

const int CONSTVECSIZE_27 = 27;
const int CONSTVECSIZE_125 = 125;
const int CONSTVECSIZE_343 = 343;
const int CONSTVECSIZE_729 = 729;
const int CONSTVECSIZE_1331 = 1331;

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


// this function is used to shift elements along the Z local column
inline __device__ void advance(float *field, const int num_points)
{
#pragma unroll
  for(int i=0; i<num_points; i++)
    field[i] = field[i+1];
}


inline __device__ void sort_bubble(float *x, int n_size)
{
	for (int i = 0; i < n_size - 1; i++)
	{
		for(int j = 0; j < n_size - i - 1; j++)
		{
			if (x[j] > x[j+1])
			{
				float temp = x[j];
				x[j] = x[j+1];
				x[j+1] = temp;
			}
		}
	}
}


// __device__  void quicksort_float(float *x, int first, int last)
// {
//    int i, j, pivot;
//    float temp;

//    if(first<last){
//       pivot=first;
//       i=first;
//       j=last;

//       while(i<j){
//          while(x[i]<=x[pivot]&&i<last)
//             i++;
//          while(x[j]>x[pivot])
//             j--;
//          if(i<j){
//             temp=x[i];
//             x[i]=x[j];
//             x[j]=temp;
//          }
//       }

//       temp=x[pivot];
//       x[pivot]=x[j];
//       x[j]=temp;
//       quicksort_float(x,first,j-1);
//       quicksort_float(x,j+1,last);

//    }
//    return;
// }

// __device__ void sort_bubble_uint16(unsigned short *x, int n_size)
// {
// 	for (int i = 0; i < n_size - 1; i++)
// 	{
// 		for(int j = 0; j < n_size - i - 1; j++)
// 		{
// 			if (x[j] > x[j+1])
// 			{
// 				unsigned short temp = x[j];
// 				x[j] = x[j+1];
// 				x[j+1] = temp;
// 			}
// 		}
// 	}
// }

// __device__ void sort_linear(float *x, int n_size)
// {
// 	for (int i = 0; i < n_size-1; i++)
// 	{
// 		int min_idx = i;
// 		for (int j = i + 1; j < n_size; j++)
// 		{
// 			if(x[j] < x[min_idx])
// 				min_idx = j;
// 		}
// 		float temp = x[min_idx];
// 		x[min_idx] = x[i];
// 		x[i] = temp;
// 	}
// }

/*shared macros*/
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const {
            return (float)(x*x);
        }
};

/*checks CUDA call, should be used in functions returning <int> value
if error happens, writes to standard error and explicitly returns -1*/
#define checkCudaErrors(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        return -1;                                                                \
    }                                                                          \
}
