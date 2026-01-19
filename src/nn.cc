#include "nn.h"
#include "encoder.h"
#include <chrono>

NeuralNetwork::NeuralNetwork(const std::string& model_path, int b_size) 
  : device(torch::kCUDA), batch_size(b_size), running(true) 
{ 
        try {
                module = torch::jit::load(model_path);
                module.to(device);
                module.eval();
                batcher_thread = std::thread(&NeuralNetwork::batcher_loop, this);
        } catch (const c10::Error& e) {
                std::cerr << "Error loading the model: " << e.msg() << std::endl;
        }
}

NeuralNetwork::~NeuralNetwork() 
{
        running = false;
        cv.notify_all();
        if (batcher_thread.joinable()) batcher_thread.join();
}

EncodedState NeuralNetwork::get_encoded_state_static(const GameState &state) 
{
        return encode_state(state);
}

std::future<EvalResult> NeuralNetwork::evaluate_async(const GameState &state) 
{
        std::promise<EvalResult> promise;
        auto future = promise.get_future();
        
        EncodedState encoded = encode_state(state);
        
        {
                std::lock_guard<std::mutex> lock(queue_mtx);
                queue.push({encoded, std::move(promise)});
        }
        
        cv.notify_one();
        return future;
}

void NeuralNetwork::batcher_loop() 
{
        while (running) {
                std::vector<Request> batch;
                
                {
                        std::unique_lock<std::mutex> lock(queue_mtx);
                        cv.wait(lock, [this] { return !queue.empty() || !running; });
                        
                        if (!running && queue.empty()) break;
                        
                        // Drain queue
                        while (!queue.empty() && batch.size() < (size_t)batch_size) {
                                batch.push_back(std::move(queue.front()));
                                queue.pop();
                        }
                }
                
                
                
                try {
                        // Prepare Batch Tensor
                        std::vector<float> flat_input;
                        flat_input.reserve(batch.size() * 48 * 9 * 9);
                        for (const auto& req : batch) 
                                flat_input.insert(flat_input.end(), req.state.begin(), req.state.end());
                        
                        
                        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
                        torch::Tensor input_tensor = torch::from_blob(flat_input.data(), 
                                {(long)batch.size(), 48, 9, 9}, opts).clone().to(device);

                        // Inference

                        auto outputs = module.forward({input_tensor}).toTuple();
                        at::Tensor p_batch = outputs->elements()[0].toTensor().softmax(1).cpu();
                        at::Tensor v_batch = outputs->elements()[1].toTensor().cpu();
                        
                        auto p_accessor = p_batch.accessor<float, 2>();
                        auto v_accessor = v_batch.accessor<float, 2>();
                        
                        for (size_t i = 0; i < batch.size(); ++i) {
                                std::vector<float> policy(13932);
                                for(int j=0; j<13932; ++j) policy[j] = p_accessor[i][j];
                                
                                batch[i].promise.set_value({policy, v_accessor[i][0]});
                        }
                } catch (...) {
                        std::cerr << "Inference failed!" << std::endl;
                                        // CRITICAL: Notify workers so they don't hang
                            for (auto& req : batch) {
                                // You could also use set_exception here
                                req.promise.set_value({std::vector<float>(13932, 0.0f), 0.0f}); 
                            }
                }
        }
}
