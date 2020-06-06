#include <unordered_map>
#include <utility>

#include "Feature.hpp"
#include "FeatureVisitor.hpp"

/* PROCESSING VISITOR */

ProcessingFeatureVisitor::ProcessingFeatureVisitor(MemoryAllocationFeatureVisitor& mem_visitor): 
    FeatureVisitor(),
    memory(std::move(mem_visitor.moveAllocated())) { }

// For now, I am leaving the implementations blank.
void ProcessingFeatureVisitor::dispatch(NumericFeature<float>* feature) { }
void ProcessingFeatureVisitor::dispatch(NumericFeature<double>* feature) { }
void ProcessingFeatureVisitor::dispatch(BertFeature* feature) { }
void ProcessingFeatureVisitor::dispatch(MemoryMappedLookupTableFeature* feature) { }

/* MEMORY ALLOCATION VISITOR */

MemoryAllocationFeatureVisitor::MemoryAllocationFeatureVisitor(size_t nrows_): 
    FeatureVisitor(),
    nrows(nrows_),
    memory(std::make_unique<HashMap<String,UPtr<Bytes>>>()) { }

void MemoryAllocationFeatureVisitor::dispatch(NumericFeature<double>* feature) {
    this->dispatchImpl(feature);
}
void MemoryAllocationFeatureVisitor::dispatch(NumericFeature<float>* feature) {
    this->dispatchImpl(feature);
}
void MemoryAllocationFeatureVisitor::dispatch(BertFeature* feature) {
    this->dispatchImpl(feature);
}
void MemoryAllocationFeatureVisitor::dispatch(MemoryMappedLookupTableFeature* feature) {
    return;
}

UPtr<HashMap<String, UPtr<Bytes>>> MemoryAllocationFeatureVisitor::moveAllocated(void) {
    return std::move(this->memory);
}

/* COMPOSITE FEATURE VISITOR */

CompositeFeatureVisitor::CompositeFeatureVisitor(UPtr<std::vector<UPtr<FeatureVisitor>>> subvisitors) {
    this->subvisitors = std::move(subvisitors); 
}

void CompositeFeatureVisitor::dispatch(NumericFeature<double>* feature) {
    for (auto& visitor: *subvisitors) visitor->dispatch(feature);
}
void CompositeFeatureVisitor::dispatch(NumericFeature<float>* feature) {
    for (auto& visitor: *subvisitors) visitor->dispatch(feature);
}
void CompositeFeatureVisitor::dispatch(BertFeature* feature) {
    for (auto& visitor: *subvisitors) visitor->dispatch(feature);
}
void CompositeFeatureVisitor::dispatch(MemoryMappedLookupTableFeature* feature) {
    for (auto& visitor: *subvisitors) visitor->dispatch(feature);
}
