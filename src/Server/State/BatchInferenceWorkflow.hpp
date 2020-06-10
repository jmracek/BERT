#ifndef INFERENCEWORKFLOW_HPP
#define INFERENCEWORKFLOW_HPP

#include "../Message/BatchInferenceRequest.capnp.h"
#include <memory>
/* The code you'll find here is telling our server what to do when it receives
 * a request to do BatchInference.  We use a hybrid combination of State, Flywheel, 
 * and Visitor patterns.
 */
class WorkflowDispatcher;
class BatchInferenceWorkflowDispatcher;
class BatchInferenceRequestState;

class ThreadPool;
class ModelStore;
class Model;
class ModelConfig;
class ModelInput;
class ModelOutput;
class FeatureSpec;
class FeatureVisitor;

// Later: The type of the model, i.e. MXNetModel, CUDAModel, etc. determines adapters for the input and output.
class BatchInferenceWorkflow {
private:
    friend class BatchInferenceWorkflowDispatcher;
    void changeState(BatchInferenceRequestState& state);
    
    int sock_fd;
    std::shared_ptr<BatchInferenceRequest::Reader> rd;
    BatchInferenceRequestState&        state;
    std::shared_ptr<ThreadPool>         pool;
    std::shared_ptr<ModelStore>        store;
    std::shared_ptr<FeatureVisitor> fvisitor;
    std::shared_ptr<FeatureSpec>       fspec;
    std::shared_ptr<Model>             model;
    std::shared_ptr<ModelInput>       inputs;
    std::shared_ptr<ModelOutput>     outputs;
public:
    BatchInferenceWorkflow(int sock_fd); 
    void accept(BatchInferenceWorkflowDispatcher& dispatcher);
};

template<typename T>
class Singleton {
public:
    Singleton& operator= (const Singleton&) = delete;
    Singleton& operator= (Singleton&&)      = delete;
    static T& instance(void) {
        static T object;
        return object;
    }
};

class BatchInferenceRequestState {
public:
    virtual void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) = 0;
};

class RequestReceived: public BatchInferenceRequestState, public Singleton<RequestReceived> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class RequestParsed: public BatchInferenceRequestState, public Singleton<RequestParsed> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class ModelConfigurationReady: public BatchInferenceRequestState, public Singleton<ModelConfigurationReady> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class ProcessingFeatures: public BatchInferenceRequestState, public Singleton<ProcessingFeatures> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class ProcessingComplete: public BatchInferenceRequestState, public Singleton<ProcessingComplete> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class ModelInputReady: public BatchInferenceRequestState, public Singleton<ModelInputReady> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class InferenceInProgress: public BatchInferenceRequestState, public Singleton<InferenceInProgress> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class PredictionsReady: public BatchInferenceRequestState, public Singleton<PredictionsReady> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class SentClientResponse: public BatchInferenceRequestState, public Singleton<SentClientResponse> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

class BatchInferenceWorkflowFailure: public BatchInferenceRequestState, public Singleton<BatchInferenceWorkflowFailure> {
public:
    void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) override;
};

enum class BatchInferenceWorkflowError {
    UNRECOGNIZED_MESSAGE,
    MODEL_NOT_FOUND,
    FSPEC_NOT_FOUND,
    PROCESSING_ERROR, 
    INFERENCE_ERROR
};

#endif
