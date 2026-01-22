#pragma once
#include "types.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/script.h>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

// A struct to hold the result of an evaluation
struct EvalResult {
    torch::Tensor policy;
    float value;
};

class NeuralNetwork {
  public:
    NeuralNetwork(const std::string& model_path, int batch_size = 64);
    ~NeuralNetwork();

    // returns a future that will be fulfilled by the batcher
    std::future<EvalResult> evaluate_async(const GameState &state);
    
    // helper to get encoded state 
    static EncodedState get_encoded_state_static(const GameState &state);

  private:
    void batcher_loop(); // Background thread function

    torch::jit::script::Module module;
    torch::Device device;
    int queue_size;
    std::atomic<bool> running;

    // Batching Queue
    struct Request {
        EncodedState state;
        std::promise<EvalResult> promise;
    };
    std::queue<Request> queue;
    std::mutex queue_mtx;
    std::condition_variable cv;
    std::thread batcher_thread;
};
