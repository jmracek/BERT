/*
class Featurizer {
private:
    std::shared_ptr<FeatureVisitor> visitor;
public:
    Featurizer(std::shared_ptr<FeatureVisitor> visitor)
    std::shared_ptr<ModelInput> process(const Request& rq, const FeatureSpec& fspec); 
};

RETURN_TYPE Featurizer::process(const Request& rq, const FeatureSpec& fspec) {
    // Check that the schema of request is compatible with fspec?
    // Allocate memory to perform all proccessing
    // Traverse each node in the FeatureSpec

}

1. Kick off std::async jobs to process the bottom features.
2. When the job completes, there is a callback to inform parent nodes
*/


class AsyncGraphTraversalDispatcher {
private:
    HashMap<String, std::atomic<int>> completed_count;
    HashMap<String, int> descendant_count;
    const FeatureSpec& fspec;
    
    AsyncGraphTraversalDispatcher(const FeatureSpec& fspec);

    void traverse(FeatureVisitor& visitor);
};


// For each child node, kick off the processing job.
// When the job finishes,
//  a) inform all the nodes parents that this node is complete
//  b) while informing the parent, check whether the given node is
//     the last one to complete its task.
//  c) if all child nodes of that parent are finished, kick off another
//     async task to process the parent.
//  d) if there are more nodes to complete, end.
void traverse(FeatureVisitor& visitor) {
    std::vector<std::future<bool>> pending;

    auto task = [](const String& node, FeatureVisitor& visitor, AsyncGraphTraversalDispatcher* dispatcher) -> bool {
        // Perform the processing
        fspec[node]->accept(visitor);
        std::vector<String>& parents = fspec.getParents(node);

        // The Batman condition.
        if (parents.empty()) {
            return true;
        }
        else {
            for (auto& parent: parents) {
                int n_complete = ++(dispatcher->completed_count.at(parent));
                if (n_complete == dispatcher->descendent_count.at(parent)) {
                    std::async(
                        std::launch::async,
                        task,
                        parent, 
                        visitor, 
                        dispatcher
                    );
                }
            }
        }
    }
    
    for (auto& node: fspec.childNodes()) {
        pending.emplace_back(std::async(
            std::launch::async,
            task,
            node,
            parent,
            visitor,
            this 
        ));
    }
    
    bool success = true;
    for(auto& result: pending) {
        success &= result.get();
    }

    if (success) {
        return;
    }
    else {
        std::cout << "[ERROR]: Processing FeatureSpec has failed" << std::endl;
    }
}

