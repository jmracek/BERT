#include "BatchInferenceRequest.capnp.h"

class BatchInferenceRequestDeserializer {
private:
    void parse(arrow::Type::type, unsigned int n);

public:
    std::unique_ptr<BatchInferenceRequest> deserialize(Buffer* buffer) {
        
    }
};

struct BatchInferenceRequest {
    unsigned int model_id;
    unsigned int batch_size;
    std::string api_key;
    HashMap<String, SPtr<arrow::Array>> data;
    HashMap<String, arrow::Type::type> schema;
};

