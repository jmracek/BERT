#include "BatchInferenceWorkflowDispatcher.hpp"
#include "BatchInferenceWorkflow.hpp"

#include <capnp/message.h>
#include <capnp/serialize-packed.h>

#include <optional>

void BatchInferenceWorkflowDispatcher::dispatch(BatchInferenceWorkflow* wf, RequestReceived* state) {
    capnp::PackedFdMessageReader message(wf->sock_fd);
    wf->rd = std::make_shared<BatchInferenceRequest::Reader>(message.getRoot<BatchInferenceRequest>());    
    changeState(wf, SentClientResponse::instance());
}

void BatchInferenceWorkflowDispatcher::dispatch(BatchInferenceWorkflow* wf, RequestParsed* state) {
    std::optional<ModelConfig> m_config = wf->store->getModelConfig(wf->model_id);

    if (!m_config) {
        changeState(wf, BatchInferenceWorkflowFailure::instance());
    }
    
    wf->model   = m_config->getModel();
    wf->fspec   = m_config->getFeatureSpec();

    changeState(wf, ModelConfigReady::instance());
}

void BatchInferenceWorkflowDispatcher::dispatch(BatchInferenceWorkflow* wf, ModelConfigurationReady* state) {
}

void BatchInferenceWorkflowDispatcher::dispatch(BatchInferenceWorkflow* wf, ProcessingFeatures* state) {
}

void BatchInferenceWorkflowDispatcher::dispatch(BatchInferenceWorkflow* wf, ModelInputReady* state) {
}

void BatchInferenceWorkflowDispatcher::dispatch(BatchInferenceWorkflow* wf, InferenceInProgress* state) {
}

void BatchInferenceWorkflowDispatcher::dispatch(BatchInferenceWorkflow* wf, PredictionsReady* state) {
}

void BatchInferenceWorkflowDispatcher::dispatch(BatchInferenceWorkflow* wf, BatchInferenceWorkflowFailure* state) {
}



/*
RequestReceived::parseMessage(BatchInferenceWorkflow* wf) {
    // Look up the API key.  Check request rate, and quit if too high.
       
    // Get the model ID from the request data
    // std::string api_key;
    unsigned int model_id;
    auto inputs = make_unique<HashMap<String, UPtr<Bytes>>>();
    
    wf->data


    // ...get inputs from the buffer and put them into the hashmap

}

// Get the model we need to perform inference
void RequestParsed::getModelConfig(BatchInferenceWorkflow* wf) {
}

// Perform feature preprocessing
void ModelConfigReady::processFeatures(BatchInferenceWorkflow* wf) {
    AsyncGraphTraversalDispatcher dispatcher(wf->fspec, wf->pool);
    dispatcher.traverse(wf->fvisitor);
    changeState(wf, ProcessingFeatures::instance());   
}

void FeatureProcessingComplete::getResult(BatchInferenceWorkflow* wf) {
     
    changeState(wf, ModelInputReady::instance());
}

void ModelInputReady::inference(BatchInferenceWorkflow* wf) {
    wf->outputs = *(wf->model)(wf->inputs);
    changeState(wf, InferenceInProgress::instance());
}

*/
