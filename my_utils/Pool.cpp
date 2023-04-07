#include "Pool.hpp"

using namespace std;

Pool::Pool(int thread_num):_stop_event(false){
    pool.reserve(thread_num);
    for(int i=0; i<thread_num; i++){
        pool.emplace_back([this](){this->Worker();});
    }
}   

void Pool::Worker(){
    while(true){
        unique_lock<std::mutex> lock(m);
        // wait until job is added or stop event set
        cv.wait(lock, [this](){return !(this->job.empty())||this->_stop_event;});
        if(_stop_event)
            return;
        
        function<void()> j = job.front();
        job.pop();
        lock.unlock();

        // do job function
        j();
    }
}

Pool::~Pool(){
    _stop_event = true;
    cv.notify_all();
    for(auto& t: pool)
        t.join();
}