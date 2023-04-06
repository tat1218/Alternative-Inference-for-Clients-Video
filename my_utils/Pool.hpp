#pragma once

#include <iostream>
#include <thread>
#include <queue>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <type_traits>
#include <utility>

using namespace std;

class Pool{
public:
    Pool(int thread_num);
    ~Pool();
    template<class Func, class... Args>
    future<std::invoke_result_t<Func,Args...>> AddJob(Func&& f,Args&&... args){
        cout << "Job is Added" << endl;
        if(_stop_event)
            throw runtime_error("Thread Pool is stopped!");

        using return_type = typename invoke_result<Func,Args...>::type;
        auto j = make_shared<packaged_task<return_type()>>(bind(f,forward<Args>(args)...));
        future<return_type> job_result = j->get_future();
        {
            lock_guard<mutex> lock(m);
            job.push([j](){(*j)();});
        }
        cv.notify_one();

        return job_result;
    }
private:
    vector<thread> pool;
    queue<function<void()>> job;
    condition_variable cv;
    mutex m;
    bool _stop_event;
    void Worker();
};