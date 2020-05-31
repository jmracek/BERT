#include <iostream>
#include <utility>

#include "FeatureSpec.hpp"
#include "Feature.hpp"

FeatureSpec::Builder::Builder(void) {
    name_lookup = std::make_unique<HashMap<String, UPtr<Feature>>>();
    graph       = std::make_unique<HashMap<String, std::vector<String>>>();
    outputs     = std::make_unique<std::vector<String>>();
}
        
FeatureSpec FeatureSpec::Builder::build(void) { return FeatureSpec(*this); }

FeatureSpec::Builder FeatureSpec::newBuilder(void) { return Builder(); }

FeatureSpec::FeatureSpec(Builder& builder) {
    this->name_lookup = std::move(builder.name_lookup);
    this->graph = std::move(builder.graph);
    this->outputs = std::move(builder.outputs);
}
