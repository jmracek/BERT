#include <optional>

//TODO
// 1. parseMessage
// 2. Decide how to
//  a) Instantiate visitors
//  b) Get the result out of a visitor after processing is done
// 3. 


RequestReceived::parseMessage(BatchInferenceWorkflow* wf) {
    // Look up the API key.  Check request rate, and quit if too high.
       
    // Get the model ID from the request data
    // std::string api_key;
    unsigned int model_id;
    auto inputs = make_unique<HashMap<String, UPtr<Bytes>>>();
    
    wf->data


    // ...get inputs from the buffer and put them into the hashmap

    changeState(wf, RequestParsed::instance());
}

// Get the model we need to perform inference
void RequestParsed::getModelConfig(BatchInferenceWorkflow* wf) {
    std::optional<ModelConfig> m_config = wf->store->getModelConfig(wf->model_id);

    if (!m_config) {
        changeState(wf, BatchInferenceWorkflowFailure::instance());
    }
    
    wf->model   = m_config->getModel();
    wf->fspec   = m_config->getFeatureSpec();

    changeState(wf, ModelConfigReady::instance());
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
