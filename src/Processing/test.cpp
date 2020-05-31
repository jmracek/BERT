#include "FeatureSpec.hpp"
#include "Feature.hpp"

int main(void) {
    FeatureSpec fspec = FeatureSpec::newBuilder()
        .addFeature(MemoryMappedLookupTableFeature("asin_"), {"asin"})
        .addFeature(MemoryMappedLookupTableFeature("page_asin_"), {"page_asin"})
        .addFeature(BertFeature(0, 1, 2, 128, "bert_t"), {"ad_asin_title", "page_asin_title"});
}

/*
TODO:
    1) Implement concrete subclass of FeatureVisitor (SeerFeatureVisitor)
    2) Implement SentencePiece processing for BERT
    3) Implement MemoryMappedLookupTableFeature
*/

