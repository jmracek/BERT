#ifndef BATCH_INFERENCE_STATE_HPP
#define BATCH_INFERENCE_STATE_HPP

#include <memory>
#include <vector>

#include "../buffer.hpp"

class BatchInferenceStateVisitor;

class BatchInferenceRequestState {
public:
    virtual std::unique_ptr<BatchInferenceRequestState> accept(BatchInferenceStateVisitor& visitor) = 0;
    virtual bool workflowComplete(void) = 0;
};

class BatchInferenceBeginState: public BatchInferenceRequestState {
public:
    std::unique_ptr<BatchInferenceRequestState> accept(BatchInferenceStateVisitor& visitor) override;
    bool workflowComplete(void) override;
};

class BatchInferenceMessageParsed: public BatchInferenceRequestState {
public:
    std::unique_ptr<BatchInferenceRequestState> accept(BatchInferenceStateVisitor& visitor) override;
    bool workflowComplete(void) override;
};

class BatchInferenceModelInputReady: public BatchInferenceRequestState {
public:
    std::unique_ptr<BatchInferenceRequestState> accept(BatchInferenceStateVisitor& visitor) override;
    bool workflowComplete(void) override;
};

class BatchInferencePredictionsReady: public BatchInferenceRequestState {
public:
    std::unique_ptr<BatchInferenceRequestState> accept(BatchInferenceStateVisitor& visitor) override;
    bool workflowComplete(void) override;
};

class BatchInferenceComplete: public BatchInferenceRequestState {
public:
    std::unique_ptr<BatchInferenceRequestState> accept(BatchInferenceStateVisitor& visitor) override;
    bool workflowComplete(void) override;
};

class BatchInferenceStateVisitor {
public:
    virtual BatchInferenceRequestState* dispatch(BatchInferenceBeginState* state) = 0;
    virtual BatchInferenceRequestState* dispatch(BatchInferenceMessageParsed* state) = 0;
    virtual BatchInferenceRequestState* dispatch(BatchInferenceModelInputReady* state) = 0;
    virtual BatchInferenceRequestState* dispatch(BatchInferencePredictionsReady* state) = 0;
    virtual BatchInferenceRequestState* dispatch(BatchInferenceComplete* state) = 0;
};

class StandardBatchInferenceStateVisitor: public BatchInferenceStateVisitor {
protected:
    Buffer* buff;
    size_t message_length;
public:
    StandardBatchInferenceStateVisitor(Buffer*, size_t);
    BatchInferenceMessageParsed* dispatch(BatchInferenceBeginState* state) override;
    BatchInferenceModelInputReady* dispatch(BatchInferenceMessageParsed* state) override;
    BatchInferencePredictionsReady* dispatch(BatchInferenceModelInputReady* state) override;
    BatchInferenceComplete* dispatch(BatchInferencePredictionsReady* state) override;
    BatchInferenceRequestState* dispatch(BatchInferenceComplete* state) override;
    size_t getMessageLength(void);
    Buffer* getBuffer(void);
};

// States
// BatchInferenceBeginState: Action - read protobuf object from request data.  check that number of inputs agrees with batchSize for each field
//  -> buffer
//  -> message len
// BatchInferenceMessageParsed: Action - process the protobuf object using the preprocessing to prepare model inputs
//  -> BatchInference message
// BatchInferenceModelInputReady: Action - Retrieve the correct model and execute model.forward().  I think I need to handle concurrency here using cuda streams?
// BatchInferencePredictionsReady: Action - Create a protobuf message out of the completed predictions
// BatchInferenceComplete
//  -> Response proto
//  -> Response len

#endif



