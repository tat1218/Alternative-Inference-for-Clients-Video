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

torch::Tensor getTensorFromImage(cv::Mat &frame, int imgSize){
    cv::Mat boxedFrame, foreground_mask, kernel=cv::Mat({3,3});
    vector<vector<cv::Point>> contours;
    cv::Rect rect;
    auto backgroundObject = cv::createBackgroundSubtractorMOG2(1000,128,false);
    bool detected;

    cv::resize(frame,frame,{854,480},0,0,1);
    cv::resize(frame,frame,{imgSize,imgSize},0,0,1);
    return torch::from_blob(frame.data,{frame.rows,frame.cols,frame.channels()},torch::kByte);
}

int main(int argc, char** argv){
    string hostname = argv[1], dataPath = argv[3];
    int portNum = atoi(argv[2]), imgSize = atoi(argv[4]), sockfd = makeConnection(hostname,portNum), n;
    sockaddr_in peerAddr;
    socklen_t addrSize = sizeof(sockaddr);
    getpeername(sockfd,(sockaddr*)&peerAddr,&addrSize);
    char *peerName = inet_ntoa(peerAddr.sin_addr);

    auto vid = cv::VideoCapture(dataPath);
    if(!vid.isOpened())
        error("ERROR opening video file");
    
    cv::Mat frame;
    torch::Tensor tensor;
    unsigned char buffer[BUF_SIZE];

    while(true){
        vid.read(frame);
        if(frame.empty()){
            error("ERROR: blank frame!");
            break;
        }
        tensor = getTensorFromImage(frame, imgSize);
        n = sendTensor(sockfd,tensor,buffer);
        if(n<0)
            error("ERROR send tensor");
        cout << "send to : " << peerName << endl;
        n = recvTensor(sockfd,tensor,buffer);
        if(n<0)
            error("ERROR recv tensor");
        cout << "recv from : " << peerName << endl;
        sleep((rand()%100)/10);
    }
    close(sockfd);
    return 0;
}