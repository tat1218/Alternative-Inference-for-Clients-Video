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
#include "my_utils/Transfer.hpp"

#define BUF_SIZE 224*224*3
#define THREAD_NUM 4

using namespace std;

void error(const char *msg){
    perror(msg);
    exit(1);
}

// DNN service for client
void action(int sockfd, string modelPath){
    int n;
    // for input, output
    long classNum;
    float classProb;
    torch::Tensor tensor, output;

    // for socket
    sockaddr_in peerAddr;
    socklen_t addrSize = sizeof(sockaddr);
    getpeername(sockfd,(sockaddr*)&peerAddr,&addrSize);
    char *peerName = inet_ntoa(peerAddr.sin_addr);
    float buffer[BUF_SIZE];

    // load DNN model from modelPath
    torch::jit::script::Module module;
    try{
        module = torch::jit::load(modelPath);
        module.to(c10::DeviceType::CPU);
        module.eval();
    }
    catch(const c10::Error& e){
        error("ERROR load model");
        return;
    }

    while(true){
        // reset
        tensor.reset();
        output.reset();
        bzero(buffer,BUF_SIZE);

        // receive tensor from client
        n = recvTensor(sockfd,tensor,buffer);
        if(n<0){
            cout << "recv end" << '\n';
            cout << "close connection to " << peerName << '\n';
            break;
        }
        cout << "recv from : " << peerName << '\n';

        // preprocess tensor and apply DNN model
        tensor.unsqueeze_(0);
        output = torch::softmax(module.forward({tensor}).toTensor(),1);

        // processing result
        tuple<torch::Tensor, torch::Tensor> result = torch::max(output,1);
        classProb = get<0>(result).accessor<float,1>()[0];
        classNum = get<1>(result).accessor<long,1>()[0];

        // send result to client
        n = write(sockfd,&classProb,sizeof(float));
        if(n<0){
            error("ERROR: send tensor");
            exit(1);
        }
        n = write(sockfd,&classNum,sizeof(float));
        if(n<0){
            error("ERROR: send tensor");
            exit(1);
        }
        cout << "send to : " << peerName << " / class : " << classNum << '\n';
    }
    close(sockfd);
    return;
}

int main(int argc, char** argv){
    // for DNN model
    string modelName = argv[2];
    // for multithreading
    Pool pool(THREAD_NUM);
    // for socket
    int sockfd, portno=atoi(argv[1]);
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    int n;
    if(argc < 2){
        fprintf(stderr,"ERROR, no port\n");
        exit(1);
    }
    // setting socket
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
        // for each client, make new socket and thread
        int newsockfd = accept(sockfd,(struct sockaddr *) &cli_addr, &clilen);
        if(newsockfd < 0)
            error("ERROR on accept");
        pool.AddJob(action,newsockfd,modelName);
    }

    close(sockfd);
    return 0;
}