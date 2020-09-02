#include <arrow/builder.h>
#include <functional>
#include <iostream>
#include <utility>

#include "FeatureSpec.hpp"
#include "Feature.hpp"

FeatureSpec::Builder::Builder(void) {
    name_lookup = std::make_unique<HashMap<String, UPtr<Feature>>>();
    op_graph    = std::make_unique<HashMap<String, std::vector<String>>>();
    outputs     = std::make_unique<std::vector<String>>();
}
        
FeatureSpec FeatureSpec::Builder::build(void) { return FeatureSpec(*this); }

FeatureSpec::Builder FeatureSpec::newBuilder(void) { return Builder(); }

FeatureSpec::FeatureSpec(Builder& builder) {
    this->name_lookup = std::move(builder.name_lookup);
    this->op_graph = std::move(builder.op_graph);
    this->outputs = std::move(builder.outputs);
}

std::vector<String>& FeatureSpec::getParents(String name) {
    return this->op_graph->at(name);
}

// This function is allocating space to do any feature processing.
// Each feature has a corresponding type and shape.
// How will this be consumed by other elements?
/*
HashMap<String, void *> FeatureSpec::alloc(size_t nrows) {
    HashMap<String, std::unique_ptr<arrow::ArrayBuilder>> output;
    void *ptr;
    for (auto& [name, feature]: *name_lookup) {
         
        switch(feature.type()) {
        case Seer::Type::INT32:
                            
        case Seer::Type::INT64:
        case Seer::Type::FP16:
        case Seer::Type::FP32:
        case Seer::Type::FP64:
        default:
        }
         




        std::vector<size_t> shape = feature.shape()
        size_t num_elements = std::accumulate(
            shape.begin(), 
            shape.end(), 
            nrows,
            [](size_t Acc, size_t x) {
                return Acc * x;
            }
        );
        
    }
}*/
