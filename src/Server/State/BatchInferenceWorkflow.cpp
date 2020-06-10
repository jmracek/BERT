#include "BatchInferenceWorkflow.hpp"
#include "BatchInferenceWorkflowDispatcher.hpp"

#include <memory>

BatchInferenceWorkflow::BatchInferenceWorkflow(int sock_fd):
    state(RequestReceived::instance()) { 
    this->sock_fd = sock_fd; 
}

void BatchInferenceWorkflow::accept(BatchInferenceWorkflowDispatcher& dispatcher) {
    this->state.accept(dispatcher, this);
}

void BatchInferenceWorkflow::changeState(BatchInferenceRequestState& new_state) {
    this->state = new_state;
}


void RequestReceived::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}

void RequestParsed::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}

void ModelConfigurationReady::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}

void ProcessingFeatures::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}

void ProcessingComplete::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}

void InferenceInProgress::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}

void PredictionsReady::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}

void SentClientResponse::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}

void BatchInferenceWorkflowFailure::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, *this);
}
