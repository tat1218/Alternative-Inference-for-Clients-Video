# Video Inference Instaed of Client(VIIC)

* server-client connection model
* client send video to server, then receive inference result
* server receive each frame and apply DNN model to frame, then send result to client

## File Description

* [my_utils/Pool.hpp, my_utils/Pool.cpp] - Thread pool class used in server
* [my_utils/Transfer.hpp] - functions for sending and receiving tensor between server and client
* [server.cpp] - connection setup for client and provide client with DNN inference service
* [client.cpp] - connect to server ,then send video frame in tensor form

## environment
* linux 20.04 LTS
* GCC 9.4.0
* CMake 3.16.3
* Libtorch 2.0.0 (CPU version)
* opencv 4.4.0

## How to run
#### 1. Build binary files
>  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch
>  cmake CMakeLists.txt
>  make

#### 2. Run server
>  ./server [port_number] [/path/to/model]

#### 3. Run client
>  ./client [server_address] [port_number] [/path/to/video] [model_image_size]
