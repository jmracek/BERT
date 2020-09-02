#ifndef WORKFLOW_TRAVERSE_HPP
#define WORKFLOW_TRAVERSE_HPP

#include <atomic>
#include <memory>
#include <unordered_map>

class Workflow;
class ThreadPool;
template<typename Task> class TaskVisitor;

class AsyncWorkflowTraversalDispatcher {
private:
    HashMap<String, std::atomic<int>> remaining_count;
    std::shared_ptr<ThreadPool>   pool;
    std::shared_ptr<Workflow> workflow;
public:
    AsyncWorkflowTraversalDispatcher(std::shared_ptr<Workflow> wf, std::shared_ptr<ThreadPool> p);

    template<typename Task>
    void traverse(std::shared_ptr<TaskVisitor<Task>> visitor);
};

#endif


