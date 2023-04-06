# DNN-server-module
-server-client connection model\n
-client send video to server, then receive inference result.\n
-server receive each frame and apply DNN model to frame, then send result to client.\n

# environment
- opencv 4.4.0
- libtorch
- linux 20.04

# run
$ cmake CMakeLists.txt
$ make
