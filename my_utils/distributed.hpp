#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <torch/torch.h>

ssize_t sendTensor(int sockfd, torch::Tensor tensor, unsigned char *buffer){
    size_t dim = tensor.sizes().size();
    int n = write(sockfd, &dim, sizeof(size_t));
    if(n<0)
        return n;
    int64_t tensorShape[dim], tensorSize=1;
    for(int i=0; i<dim; i++){
        tensorShape[i] = tensor.sizes()[i];
        tensorSize *= tensorShape[i];
    }
    n = write(sockfd, tensorShape, sizeof(int64_t)*dim);
    if(n<0)
        return n;
    
    buffer = tensor.data_ptr<unsigned char>();
    n = write(sockfd,buffer,tensorSize);
    std::cout << "send tensor : " << tensorSize << std::endl;
    return n;
}

ssize_t recvTensor(int sockfd,torch::Tensor &tensor, unsigned char *buffer){
    size_t dim;
    int n = read(sockfd,&dim,sizeof(size_t));
    if(n<0)
        return n;
    int64_t tensorShape[dim], tensorSize=1;
    n = read(sockfd,tensorShape, sizeof(int64_t)*dim);
    if(n<0)
        return n;

    for(int i=0; i<dim; i++)
        tensorSize *= tensorShape[i];
    n = read(sockfd,buffer,tensorSize);
    if(n<0)
        return n;

    auto shape = c10::ArrayRef<int64_t>(tensorShape,dim);
    auto options = torch::TensorOptions().dtype(torch::kByte);
    tensor = torch::from_blob(buffer,{1,shape[2],shape[0],shape[1]},options);
    std::cout << "recv tensor : " << tensorSize << std::endl;
    return n;
}