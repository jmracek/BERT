
/* The code you'll find here is telling our server what to do when it receives
 * a request to do BatchInference.  We use a hybrid combination of State, Flywheel, 
 * and Visitor patterns.
 */

// Later: The type of the model, i.e. MXNetModel, CUDAModel, etc. determines adapters for the input and output.
class BatchInferenceWorkflow {
private:
    friend class BatchInferenceRequestState;
    void changeState(BatchInferenceRequestState* state);
    
    int sock_fd;
    std::shared_ptr<BatchInferenceRequest::Reader> rd;
    std::shared_ptr<BatchInferenceRequestState> state;
    std::shared_ptr<ThreadPool>         pool;
    std::shared_ptr<Buffer>             data;
    std::shared_ptr<ModelStore>        store;
    std::shared_ptr<FeatureVisitor> fvisitor;
    std::shared_ptr<FeatureSpec>       fspec;
    std::shared_ptr<Model>             model;
    std::shared_ptr<ModelInput>       inputs;
    std::shared_ptr<ModelOutput>     outputs;
public:
    BatchInferenceWorkflow(int sock_fd); 
    void accept(WorkflowDispatcher& dispatcher);
};


class BatchInferenceRequestState {
protected:
    void changeState(BatchInferenceWorkflow* wf, std::shared_ptr<BatchInferenceRequestState> new_state);
public:
    virtual void accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) = 0;
};

class RequestReceived;
class RequestParsed;
class ModelConfigReady;
class ProcessingFeatures;
class ProcessingComplete;
class ModelInputReady;
class InferenceInProgress;
class PredictionsReady;
class SentClientResponse;
class BatchInferenceWorkflowFailure;


enum class BatchInferenceWorkflowError {
    UNRECOGNIZED_MESSAGE,
    MODEL_NOT_FOUND,
    FSPEC_NOT_FOUND,
    PROCESSING_ERROR, 
    INFERENCE_ERROR
};


    
/*
    void parseMessage();
    void getModelConfig();
    void processFeatures();
    void inference();
    void getResult();
    void getError();
*/


class WorkflowDispatcher {
private:
    bool _done = false;
public:
    virtual void dispatch(BatchInferenceWorkflow* wf, RequestReceived* state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, RequestParsed* state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, ModelConfigurationReady* state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, ProcessingFeatures* state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, ModelInputReady* state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, InferenceInProgress* state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, PredictionsReady* state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, BatchInferenceWorkflowFailure* state) = 0;
    bool finished(void) { return this->_done; }
};
