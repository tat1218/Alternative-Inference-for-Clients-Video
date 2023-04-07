#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <thread>
#include "my_utils/Pool.hpp"
#include "my_utils/distributed.hpp"

#define BUF_SIZE 224*224*3
#define THREAD_NUM 4

using namespace std;

void error(const char *msg){
    perror(msg);
    exit(1);
}

void action(int sockfd, string modelPath){
    int n;
    at::Tensor tensor;
    at::Tensor output;// = torch::ones({1,3,224,224},c10::TensorOptions().dtype(torch::kFloat32));
    sockaddr_in peerAddr;
    socklen_t addrSize = sizeof(sockaddr);
    getpeername(sockfd,(sockaddr*)&peerAddr,&addrSize);
    char *peerName = inet_ntoa(peerAddr.sin_addr);
    unsigned char buffer[BUF_SIZE]; 
    torch::jit::script::Module module;
    try{
        module = torch::jit::load(modelPath);
        //module.to(at::kCUDA);
        module.eval();
    }
    catch(const c10::Error& e){
        error("ERROR load model");
        return;
    }

    while(true){
        bzero(buffer,BUF_SIZE);
        n = recvTensor(sockfd,tensor,buffer);
        if(n<0){
            error("ERROR: recv tensor");
            exit(1);
        }
        tensor = tensor.permute({2,0,1}).to(at::kFloat);
        tensor = tensor.reshape({1,tensor.sizes()[0],tensor.sizes()[1],tensor.sizes()[2]});
        cout << "recv from : " << peerName << endl;
        //cout << "tensor : " << tensor << endl;
        output = module.forward({tensor}).toTensor();
        
        cout << "output : " ;//<< output << endl;
        //bzero(buffer,BUF_SIZE);
        
        n = sendTensor(sockfd,output,buffer);
        cout << "send to : " << peerName << endl;
        if(n<0){
            error("ERROR: send tensor");
            exit(1);
        }
        tensor.reset();
        output.reset();
    }
    std::cout << "recv end" << endl;
    close(sockfd);
    return;
}

int main(int argc, char** argv){
    int sockfd, portno=atoi(argv[1]);
    string modelName = argv[2];
    Pool pool(THREAD_NUM);

    pid_t pid;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    int n;
    if(argc < 2){
        fprintf(stderr,"ERROR, no port\n");
        exit(1);
    }
    sockfd = socket(AF_INET,SOCK_STREAM,0);
    if(sockfd<0)
        error("EROR opening socket");
    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if(bind(sockfd, (struct sockaddr *) &serv_addr,sizeof(serv_addr))<0)
        error("ERROR on binding");
    listen(sockfd,THREAD_NUM);
    clilen = sizeof(cli_addr);
    
    while(1){
        int newsockfd = accept(sockfd,(struct sockaddr *) &cli_addr, &clilen);
        if(newsockfd < 0)
            error("ERROR on accept");
        pool.AddJob(action,newsockfd,modelName);
    }

    close(sockfd);
    return 0;
}