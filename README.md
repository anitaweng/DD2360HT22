# DD2360HT22

Exectution commands for ```ex_2``` using the commands below:
```
export PATH=/usr/local/cuda/bin:$PATH
nvcc -arch=sm_50 -I${path_to_cuda_sample_files/Common} deviceQuery.cpp -o deviceQuery
```
Change ```${path_to_cuda_sample_files/Common}``` to your own path.
