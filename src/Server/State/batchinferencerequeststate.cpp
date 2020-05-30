#include <memory>
#include <iostream>

#include "batchinferencerequeststate.hpp"

std::unique_ptr<BatchInferenceRequestState> BatchInferenceBeginState::accept(BatchInferenceStateVisitor& visitor) {
    return std::unique_ptr<BatchInferenceRequestState>(visitor.dispatch(this));
}

std::unique_ptr<BatchInferenceRequestState> BatchInferenceMessageParsed::accept(BatchInferenceStateVisitor& visitor) {
    return std::unique_ptr<BatchInferenceRequestState>(visitor.dispatch(this));
}

std::unique_ptr<BatchInferenceRequestState> BatchInferenceModelInputReady::accept(BatchInferenceStateVisitor& visitor) {
    return std::unique_ptr<BatchInferenceRequestState>(visitor.dispatch(this));
}

std::unique_ptr<BatchInferenceRequestState> BatchInferencePredictionsReady::accept(BatchInferenceStateVisitor& visitor) {
    return std::unique_ptr<BatchInferenceRequestState>(visitor.dispatch(this));
}

std::unique_ptr<BatchInferenceRequestState> BatchInferenceComplete::accept(BatchInferenceStateVisitor& visitor) {
    return std::unique_ptr<BatchInferenceRequestState>(visitor.dispatch(this));
}

bool BatchInferenceBeginState::workflowComplete(void) {
    return false;
}

bool BatchInferenceMessageParsed::workflowComplete(void) {
    return false;
}

bool BatchInferenceModelInputReady::workflowComplete(void) {
    return false;
}

bool BatchInferencePredictionsReady::workflowComplete(void) {
    return false;
}

bool BatchInferenceComplete::workflowComplete(void) {
    return true;
}

StandardBatchInferenceStateVisitor::StandardBatchInferenceStateVisitor(Buffer* buff, size_t message_length) {
    this->buff = buff;
    this->message_length = message_length;
}

BatchInferenceMessageParsed* StandardBatchInferenceStateVisitor::dispatch(BatchInferenceBeginState* state) {
    std::cout << "Begin State" << std::endl;
    return new BatchInferenceMessageParsed();
}

BatchInferenceModelInputReady* StandardBatchInferenceStateVisitor::dispatch(BatchInferenceMessageParsed* state) {
    std::cout << "Message Parsed" << std::endl;
    return new BatchInferenceModelInputReady();
}

BatchInferencePredictionsReady* StandardBatchInferenceStateVisitor::dispatch(BatchInferenceModelInputReady* state) {
    std::cout << "Model Input Ready" << std::endl;
    return new BatchInferencePredictionsReady();
}

BatchInferenceComplete* StandardBatchInferenceStateVisitor::dispatch(BatchInferencePredictionsReady* state) {
    std::cout << "Predictions Ready" << std::endl;
    return new BatchInferenceComplete();
}

BatchInferenceRequestState* StandardBatchInferenceStateVisitor::dispatch(BatchInferenceComplete* state) {
    std::cout << "Complete!" << std::endl;
    return state;
}

size_t StandardBatchInferenceStateVisitor::getMessageLength(void) {
    return this->message_length;
}

Buffer* StandardBatchInferenceStateVisitor::getBuffer(void) {
    return this->buff;
}
