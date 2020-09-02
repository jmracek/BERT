#ifndef WORKFLOW_HPP
#define WORKFLOW_HPP

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class WorkflowTask;

template<typename T>
using UPtr = std::unique_ptr<T>;

template<typename K, typename V>
using HashMap = std::unordered_map<K,V>;

using String = std::string;

class Workflow {
private:
    UPtr<HashMap<String, UPtr<WorkflowTask>>> name_lookup;
    UPtr<HashMap<String, std::vector<String>>> op_graph;
    UPtr<std::vector<String>> outputs;
    UPtr<std::vector<String>> child_nodes;
    
    class Builder {
    friend class Workflow;
    private:
        UPtr<HashMap<String, UPtr<WorkflowTask>>> name_lookup;
        UPtr<HashMap<String, std::vector<String>>> op_graph;
        UPtr<std::vector<String>> outputs;
        UPtr<std::vector<String>> child_nodes;
    public:
        Builder();

        template<typename ConcreteTask>
        Builder& addTask(ConcreteTask&& feature, std::vector<String> dependency_names, bool is_output = true) {
            String name = feature.getName();
            // Check that the name is unique
            if (name_lookup->find(name) != name_lookup->end()) {
                std::cout << "[ERROR]: Duplicate name \"" 
                          << name 
                          << "\" encountered when adding Task to Workflow.  Names must be unique!" 
                          << std::endl;
                exit(1);
            }

            name_lookup->insert( {name, std::make_unique<ConcreteTask>(std::forward<ConcreteTask>(feature))} );

            (*op_graph)[name] = std::vector<String>(); 
            for (const auto& child_name: dependency_names) {
                (*op_graph)[child_name].emplace_back(name);
            }

            if (is_output) outputs->push_back(name);

            return *this;
        }

        Workflow build(void);
    };

public:
    static Builder newBuilder(void);
    Workflow(Builder& builder);
    std::vector<String>& getParents(String name);
    std::vector<String>& childNodes(void);
    
    Type& operator[] (const String& name);
};

#endif
