#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <torch/torch.h>

ssize_t sendTensor(int &sockfd, torch::Tensor &tensor, float *buffer){
    // send dimension
    size_t dim = tensor.sizes().size();
    int n = write(sockfd, &dim, sizeof(size_t)); 
    if(n<0)
        return n;

    // send shape and size
    int64_t tensorShape[dim], tensorSize=1;
    for(int i=0; i<dim; i++){
        tensorShape[i] = tensor.sizes()[i];
        tensorSize *= tensorShape[i];
    }
    n = write(sockfd, tensorShape, sizeof(int64_t)*dim);
    if(n<0)
        return n;
    // send tensor
    buffer = tensor.data_ptr<float>();
    n = write(sockfd,buffer,tensorSize*sizeof(float));
    return n;
}

ssize_t recvTensor(int &sockfd,torch::Tensor &tensor, float *buffer){
    // recv dim
    size_t dim;
    int n = read(sockfd,&dim,sizeof(size_t));
    if(n<0)
        return n;
    if(dim==0)
        // for terminate
        return -1;
    
    //recv shape and size
    int64_t tensorShape[dim], tensorSize=1;
    n = read(sockfd,tensorShape, sizeof(int64_t)*dim);

    if(n<0)
        return n;
    for(int i=0; i<dim; i++)
        tensorSize *= tensorShape[i];
        
    // recv tensor
    n = read(sockfd,buffer,tensorSize*sizeof(float));
    if(n<0)
        return n;
    auto shape = c10::ArrayRef<int64_t>(tensorShape,dim);
    auto options = torch::TensorOptions().dtype(at::kFloat);
    tensor = torch::from_blob(buffer,shape,options);
    return n;
}