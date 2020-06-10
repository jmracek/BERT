// Workflow type
class BatchInferenceWorkflow;

// State types
class RequestReceived;
class RequestParsed;
class ModelConfigReady;
class ProcessingFeatures;
class ProcessingComplete;
class ModelConfigurationReady;
class ModelInputReady;
class InferenceInProgress;
class PredictionsReady;
class SentClientResponse;
class BatchInferenceWorkflowFailure;

class WorkflowDispatcher {
protected:
    void changeState(BatchInferenceWorkflow* wf, BatchInferenceRequestState& new_state);
private:
    bool _done = false;
public:
    virtual void dispatch(BatchInferenceWorkflow* wf, RequestReceived& state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, RequestParsed& state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, ModelConfigurationReady& state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, ProcessingFeatures& state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, ModelInputReady& state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, InferenceInProgress& state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, PredictionsReady& state) = 0;
    virtual void dispatch(BatchInferenceWorkflow* wf, BatchInferenceWorkflowFailure& state) = 0;
    bool finished(void) { return this->_done; }
};

class BatchInferenceWorkflowDispatcher: public WorkflowDispatcher {
public:
    void dispatch(BatchInferenceWorkflow* wf, RequestReceived& state) override;
    void dispatch(BatchInferenceWorkflow* wf, RequestParsed& state) override;
    void dispatch(BatchInferenceWorkflow* wf, ModelConfigurationReady& state) override;
    void dispatch(BatchInferenceWorkflow* wf, ProcessingFeatures& state) override;
    void dispatch(BatchInferenceWorkflow* wf, ModelInputReady& state) override;
    void dispatch(BatchInferenceWorkflow* wf, InferenceInProgress& state) override;
    void dispatch(BatchInferenceWorkflow* wf, PredictionsReady& state) override;
    void dispatch(BatchInferenceWorkflow* wf, BatchInferenceWorkflowFailure& state) override;
};
/*
    void parseMessage();
    void getModelConfig();
    void processFeatures();
    void inference();
    void getResult();
    void getError();
*/
