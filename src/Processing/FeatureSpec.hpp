#ifndef FEATURESPEC_HPP
#define FEATURESPEC_HPP

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "Feature.hpp"

template<typename T>
using UPtr = std::unique_ptr<T>;

template<typename K, typename V>
using HashMap = std::unordered_map<K,V>;

using String = std::string;

class FeatureSpec {
private:
    UPtr<HashMap<String, UPtr<Feature>>> name_lookup;
    UPtr<HashMap<String, std::vector<String>>> op_graph;
    UPtr<std::vector<String>> outputs;
    UPtr<std::vector<String>> child_nodes;
    
    class Builder {
    friend class FeatureSpec;
    private:
        UPtr<HashMap<String, UPtr<Feature>>> name_lookup;
        UPtr<HashMap<String, std::vector<String>>> op_graph;
        UPtr<std::vector<String>> outputs;
        UPtr<std::vector<String>> child_nodes;
    public:
        Builder();

        template<typename ConcreteFeature>
        Builder& addFeature(ConcreteFeature&& feature, std::vector<String> dependency_names, bool is_output = true) {
            String name = feature.getName();
            // Check that the name is unique
            if (name_lookup->find(name) != name_lookup->end()) {
                std::cout << "[ERROR]: Duplicate name \"" 
                          << name 
                          << "\" encountered when adding Feature to FeatureSpec.  Names must be unique!" 
                          << std::endl;
                exit(1);
            }

            name_lookup->insert( {name, std::make_unique<ConcreteFeature>(std::forward<ConcreteFeature>(feature))} );

            (*op_graph)[name] = std::vector<String>(); 
            for (const auto& child_name: dependency_names) {
                (*op_graph)[child_name].emplace_back(name);
            }

            if (is_output) outputs->push_back(name);

            return *this;
        }

        FeatureSpec build(void);
    };

public:
    static Builder newBuilder(void);
    FeatureSpec(Builder& builder);
    std::vector<String>& getParents(String name);
   // HashMap<String, void *> alloc(size_t nrows);
};

#endif
