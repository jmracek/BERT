#ifndef FEATUREVISITOR_HPP
#define FEATUREVISITOR_HPP

#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>

using Bytes = unsigned char[];
using String = std::string;
template<typename T>
using UPtr = std::unique_ptr<T>;
template<typename K, typename V>
using HashMap = std::unordered_map<K,V>;

// Forward declarations of Features
class BertFeature;
class MemoryMappedLookupTableFeature;
template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
class NumericFeature;

class FeatureVisitor {
public:
    virtual void dispatch(NumericFeature<float>* feature) = 0;
    virtual void dispatch(NumericFeature<double>* feature) = 0;
    virtual void dispatch(BertFeature* feature) = 0;
    virtual void dispatch(MemoryMappedLookupTableFeature* feature) = 0;
};

class MemoryAllocationFeatureVisitor: public FeatureVisitor {
private:
    UPtr<HashMap<String, UPtr<Bytes>>> memory;
    size_t nrows;
public:
    MemoryAllocationFeatureVisitor(size_t nrows);
    
    void dispatch(NumericFeature<double>* feature) override;
    void dispatch(NumericFeature<float>* feature) override;
    void dispatch(BertFeature* feature) override;
    void dispatch(MemoryMappedLookupTableFeature* feature) override;
    
    template <typename T>
    void dispatchImpl(T* feature) {
        String name = feature->getName();
        std::vector<size_t> shape = feature->getShape();

        size_t total_rank = std::accumulate(
            shape.begin(),
            shape.end(),
            this->nrows,
            [](size_t Acc, size_t x) {
                return Acc * x;
            }
        );

        size_t n_bytes = sizeof(typename T::type) * total_rank;
        
        this->memory->insert({
            name,
            std::make_unique<Bytes>(n_bytes)
        });
    }
    UPtr<HashMap<String, UPtr<Bytes>>> moveAllocated(void);
};

class ProcessingFeatureVisitor: public FeatureVisitor {
private:
    UPtr<HashMap<String, UPtr<Bytes>>> memory;
public:
    ProcessingFeatureVisitor(MemoryAllocationFeatureVisitor& mem_visitor);
    void dispatch(NumericFeature<double>* feature) override;
    void dispatch(NumericFeature<float>* feature) override;
    void dispatch(BertFeature* feature) override;
    void dispatch(MemoryMappedLookupTableFeature* feature) override;
};

class CompositeFeatureVisitor: public FeatureVisitor {
private:
    UPtr<std::vector<UPtr<FeatureVisitor>>> subvisitors;
public:
    CompositeFeatureVisitor(UPtr<std::vector<UPtr<FeatureVisitor>>> subvisitors);
    void dispatch(NumericFeature<double>* feature) override;
    void dispatch(NumericFeature<float>* feature) override;
    void dispatch(BertFeature* feature) override;
    void dispatch(MemoryMappedLookupTableFeature* feature) override;
};

#endif
