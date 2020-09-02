#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

class FeatureVisitor;

class Feature: public WorkflowTask {
protected:
    std::string name;
    std::vector<size_t> shape;
public:
    Feature(std::string name);
    std::string getName(void);
    std::vector<size_t> getShape();
    virtual void accept(FeatureVisitor& visitor) = 0;
};

// Only enable this for arithmetic types
template<typename T, typename>
class NumericFeature: public Feature {
private:
    T fill_na_value;
    bool add_na_column;
public:
    using type = T;
    NumericFeature(std::string name, T fill_na_value, bool add_na_column);
    void accept(FeatureVisitor& visitor) override;
    T getFillNaValue(void);
    bool addNaColumn(void);
};

class MemoryMappedLookupTableFeature: public Feature {
public:
    MemoryMappedLookupTableFeature(std::string name);
    void accept(FeatureVisitor& visitor) override;
};

class BertFeature: public Feature {
private:
    int cls_token;
    int sep_token;
    int msk_token;
    int max_seq_len;
public:
    using type = uint32_t;
    BertFeature(int cls, int sep, int msk, int msl, std::string name = std::string("bert"));
    void accept(FeatureVisitor& visitor) override;
};

#endif
