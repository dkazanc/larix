#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8

#define BLKXSIZE2D 16
#define BLKYSIZE2D 16

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

__device__ void sort_quick(float *x, int left_idx, int right_idx)
{
      int i = left_idx, j = right_idx;
      float pivot = x[(left_idx + right_idx) / 2];
      while (i <= j)
      {
            while (x[i] < pivot)
                  i++;
            while (x[j] > pivot)
                  j--;
            if (i <= j) {
		  float temp;
                  temp = x[i];
                  x[i] = x[j];
                  x[j] = temp;
                  i++;
                  j--;
            }
      };
      if (left_idx < j)
            sort_quick(x, left_idx, j);
      if (i < right_idx)
            sort_quick(x, i, right_idx);
}

__device__ void sort_bubble(float *x, int n_size)
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

__device__ void sort_bubble_uint16(unsigned short *x, int n_size)
{
	for (int i = 0; i < n_size - 1; i++)
	{
		for(int j = 0; j < n_size - i - 1; j++)
		{
			if (x[j] > x[j+1])
			{
				unsigned short temp = x[j];
				x[j] = x[j+1];
				x[j+1] = temp;
			}
		}
	}
}

__device__ void sort_linear(float *x, int n_size)
{
	for (int i = 0; i < n_size-1; i++)
	{
		int min_idx = i;
		for (int j = i + 1; j < n_size; j++)
		{
			if(x[j] < x[min_idx])
				min_idx = j;
		}
		float temp = x[min_idx];
		x[min_idx] = x[i];
		x[i] = temp;
	}
}

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
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        return -1;                                                             \
    }                                                                          \
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
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
