#include "AsyncWorkflowTraversalDispatcher.hpp"
#include "Workflow.hpp"
#include "WorkflowTask.hpp"
#include "../Utils/ThreadPool.hpp"

#include <vector>
#include <future>

// For each child node, kick off the processing job.
// When the job finishes,
//  a) inform all the nodes parents that this node is complete
//  b) while informing the parent, check whether the given node is
//     the last one to complete its task.
//  c) if all child nodes of that parent are finished, kick off another
//     async task to process the parent.
//  d) if there are more nodes to complete, end.
template<typename Task>
void AsyncWorkflowTraversalDispatcher::traverse(std::shared_ptr<TaskVisitor<Task>> visitor) {
    auto task = [](
        AsyncWorkflowTraversalDispatcher* dispatcher,
        const String& node, 
        std::shared_ptr<TaskVisitor> visitor, 
        std::shared_ptr<ThreadPool> pool
    ) -> void {
        // Perform the task at that node
        dispatcher->workflow[node]->accept(visitor);

        std::vector<String>& parents = dispatcher->workflow->getParents(node);
        // The Batman condition.
        if (parents.empty()) {
            return;
        }
        else {
            for (auto& parent: parents) {
                int n_remaining = --(dispatcher->remaining_count.at(parent));
                // If this is the last child to finish, launch the parent task
                if (n_remaining == 0) {
                    pool->submit(task, dispatcher, parent, visitor, pool);
                }
            }
        }
    }
    
    for (auto& node: fspec.childNodes()) {
        this->pool->submit(task, this, node, visitor, this->pool);
    }
}
