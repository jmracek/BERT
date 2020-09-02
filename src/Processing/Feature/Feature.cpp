#include "Feature.hpp"

Feature::Feature(std::string name) { this->name = name; };
std::string Feature::getName(void) { return name; }
std::vector<size_t> Feature::getShape(void) { 
    return shape;
}

MemoryMappedLookupTableFeature::MemoryMappedLookupTableFeature(std::string name):
    Feature(name) {
}
    
BertFeature::BertFeature(
    int cls,
    int sep,
    int msk,
    int msl,
    std::string name
): Feature(name), cls_token(cls), sep_token(sep), msk_token(msk), max_seq_len(msl) {
    shape = std::vector<size_t>();
    shape.emplace_back(msl);
}

template<typename T, typename U>
NumericFeature<T, U>::NumericFeature(std::string name, T fill_na_value, bool add_na_column): Feature(name) {
    this->fill_na_value = fill_na_value;
    this->add_na_column = add_na_column;
}

template<typename T, typename U>
T NumericFeature<T, U>::getFillNaValue(void) {
    return this->fill_na_value;
}

template<typename T, typename U>
bool NumericFeature<T, U>::addNaColumn(void) {
    return this->add_na_column;
}


void MemoryMappedLookupTableFeature::accept(FeatureVisitor& visitor) {
    visitor.dispatch(this);
}

void BertFeature::accept(FeatureVisitor& visitor) {
    visitor.dispatch(this);
}

template<typename T, typename U>
void NumericFeature<T, U>::accept(FeatureVisitor& visitor) {
    visitor.dispatch(this);
}

/*
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
