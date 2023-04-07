#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <arpa/inet.h>
#include "my_utils/Pool.hpp"
#include "my_utils/distributed.hpp"

#define BUF_SIZE 224*224*3

void error(const char *msg){
    perror(msg);
    exit(1);
}

// connect server
int makeConnection(string hostname, int portNum){
    int sockfd, n;
    sockaddr_in serv_addr;
    hostent *server;

    sockfd = socket(AF_INET,SOCK_STREAM,0);
    if(sockfd<0)
        error("ERROR opening socket");
    server = gethostbyname(hostname.c_str());
    if(server==NULL){
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *)&serv_addr,sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr,(char *)&serv_addr.sin_addr.s_addr,server->h_length);
    serv_addr.sin_port = htons(portNum);
    if(connect(sockfd,(sockaddr *) &serv_addr, sizeof(serv_addr))<0)
        error("ERROR connecting");
    return sockfd;
}

// change cv::Mat to torch::Tensor
torch::Tensor getTensorFromImage(cv::Mat &frame, int imgSize){
    return torch::from_blob(frame.data,{frame.rows,frame.cols,frame.channels()},torch::kByte);
}

int main(int argc, char** argv){
    // for socket
    string hostname = argv[1], dataPath = argv[3];
    int portNum = atoi(argv[2]), imgSize = atoi(argv[4]), sockfd = makeConnection(hostname,portNum), n;
    sockaddr_in peerAddr;
    socklen_t addrSize = sizeof(sockaddr);
    getpeername(sockfd,(sockaddr*)&peerAddr,&addrSize);
    char *peerName = inet_ntoa(peerAddr.sin_addr);
    float buffer[BUF_SIZE];

    // set video
    auto vid = cv::VideoCapture(dataPath);
    if(!vid.isOpened())
        error("ERROR opening video file");
    
    // for input and output
    cv::Mat frame;
    at::Tensor tensor;
    float classProb;
    long classNum;

    while(true){
        // read each frame
        vid.read(frame);
        if(frame.empty()){
            error("ERROR: blank frame!");
            break;
        }
        cv::resize(frame,frame,{imgSize,imgSize});
        if(frame.rows*frame.cols*frame.channels()>BUF_SIZE){
            error("ERROR: frame is out of buffer size");
            break;
        }

        // get tensor from image and preprocess it
        tensor = getTensorFromImage(frame, imgSize);
        tensor = tensor.permute({2,0,1}).toType(at::kFloat);

        // send tensor to server
        n = sendTensor(sockfd,tensor,buffer);
        if(n<0)
            error("ERROR send tensor");
        cout << "send to : " << peerName << endl;

        // receive DNN inference result from server
        n = read(sockfd,&classProb,sizeof(float));
        if(n<0)
           error("ERROR recv tensor");
        n = read(sockfd,&classNum,sizeof(long));
        if(n<0)
           error("ERROR recv tensor");
        cout << "recv from : " << peerName << endl;
        cout << "result : prob - " << classProb << " class number - " << classNum << endl;
        sleep(1);
    }

    // signal server for terminate
    int end = 0;
    n = write(sockfd,&end,sizeof(int));
    close(sockfd);
    return 0;
}