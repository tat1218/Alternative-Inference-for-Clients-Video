# Video Inference Instaed of Client(VIIC)

- server-client connection model

- client send video to server, then receive inference result.

- server receive each frame and apply DNN model to frame, then send result to client.

## File Description

- [my_utils/Pool.hpp, my_utils/Pool.cpp] - Thread pool class used in server
- [my_utils/Transfer.hpp] - functions for sending and receiving tensor between server and client
- [server.cpp] - connection setup for client and provide client with DNN inference service
- [client.cpp] - connect to server ,then send video frame in tensor form

## environment
* linux 20.04 LTS
* GCC 9.4.0
* CMake 3.16.3
* Libtorch 2.0.0 (CPU version)
* opencv 4.4.0

# run
$ cmake CMakeLists.txt

$ make


## How to run
#### 1. Build C++ functions
>  python3 build_script.py build_ext --inplace 

#### 2. Run the test script
>  sh ./auto_test.sh  

It takes a few seconds for heuristic algorithms, and 1 to 5 minutes for evolutionary algorithms.  
The results are created in the ./outputs/ folder.  
