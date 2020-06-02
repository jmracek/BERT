#include "Feature.hpp"

Feature::Feature(std::string name) { this->name = name; };
std::string Feature::getName(void) { return name; }
std::vector<size_t> Feature::getShape(size_t nrows) { 
    std::vector<size_t> out = shape;
    out.insert(out.begin(), nrows);
    return out;
}

MemoryMappedLookupTableFeature::MemoryMappedLookupTableFeature(std::string name):
    Feature(name) {
}

void MemoryMappedLookupTableFeature::accept(FeatureVisitor& visitor) {
    visitor.dispatch(this);
}
    
BertFeature::BertFeature(
    int cls,
    int sep,
    int msk,
    int msl,
    std::string name
): Feature(name), cls_token(cls), sep_token(sep), msk_token(msk), max_seq_len(msl) {}

void BertFeature::accept(FeatureVisitor& visitor) {
    visitor.dispatch(this);
}

/*
void SeerFeatureVisitor::dispatch(BertFeature* feature) {
    // Get the memory we've allocated for this feature
    // For
}

class Featurizer {
private:
    std::shared_ptr<FeatureSpec> fspec;
    std::shared_ptr<FeatureVisitor> visitor;
public:
    ModelInput process(Request& rq); 
};

class ModelInput {
private:
    std::unordered_map<Input, > inputs;
};
*/
