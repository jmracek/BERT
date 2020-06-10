#include "BatchInferenceWorkflow.hpp"

#include <memory>

void BatchInferenceWorkflow::accept(WorkflowDispatcher& dispatcher) {
    this->state->accept(dispatcher, this);
}

void BatchInferenceWorkFlow::changeState(BatchInferenceRequestState& new_state) {
    this->state = new_state;
}


void RequestReceived::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}

void RequestParsed::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}

void ModelConfigReady::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}

void ProcessingFeatures::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}

void ProcessingComplete::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}

void InferenceInProgress::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}

void PredictionsReady::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}

void SentClientResponse::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}

void BatchInferenceWorkflowFailure::accept(WorkflowDispatcher& dispatcher, BatchInferenceWorkflow* wf) {
    dispatcher.dispatch(wf, this);
}
