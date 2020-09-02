#include "Workflow.hpp"
#include "WorkflowTask.hpp"
#include <functional>
#include <utility>

Workflow::Builder::Builder(void) {
    name_lookup = std::make_unique<HashMap<String, UPtr<WorkflowTask>>>();
    op_graph    = std::make_unique<HashMap<String, std::vector<String>>>();
    outputs     = std::make_unique<std::vector<String>>();
}
        
Workflow Workflow::Builder::build(void) { return Workflow(*this); }

Workflow::Builder Workflow::newBuilder(void) { return Builder(); }

Workflow::Workflow(Builder& builder) {
    this->name_lookup = std::move(builder.name_lookup);
    this->op_graph = std::move(builder.op_graph);
    this->outputs = std::move(builder.outputs);
}

std::vector<String>& Workflow::getParents(String name) {
    return this->op_graph->at(name);
}
