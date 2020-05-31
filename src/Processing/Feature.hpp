#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <string>
#include <tuple>
#include <vector>

class FeatureVisitor;

class Feature {
protected:
    std::string name;
    std::vector<size_t> shape;
public:
    Feature(std::string name);
    std::string getName(void);
    std::vector<size_t> getShape(size_t nrows);
    virtual void accept(FeatureVisitor& visitor) = 0;
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
    BertFeature(int cls, int sep, int msk, int msl, std::string name = std::string("bert"));
    void accept(FeatureVisitor& visitor) override;
};

class FeatureVisitor {
public:
    virtual void dispatch(BertFeature* feature) = 0;
    virtual void dispatch(MemoryMappedLookupTableFeature* feature) = 0;
};

class SeerFeatureVisitor: public FeatureVisitor {
public:
    void dispatch(BertFeature* feature) override;
    void dispatch(MemoryMappedLookupTableFeature* feature) override;
};

#endif
