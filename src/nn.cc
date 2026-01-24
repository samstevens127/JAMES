#include "nn.h"
#include "encoder.h"
#include <chrono>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::string& model_path,const std::string& device_str, int b_size) 
  : device_str(device_str),device(device_str), queue_size(b_size), running(false) 
{  
        module = torch::jit::load(model_path);
        module.to(device);
        module.eval();
        running = true;
        batcher_thread = std::thread(&NeuralNetwork::batcher_loop, this);
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

                        auto timeout = std::chrono::milliseconds(3);

                        cv.wait_for(lock, timeout, [this] { 
                                return queue.size() >= (size_t)queue_size; 
                        });
                        
                        while (!queue.empty() && batch.size() < (size_t)queue_size) {
                                batch.push_back(std::move(queue.front()));
                                queue.pop();
                        }
                }
                
                
                
                try {
                        std::vector<torch::Tensor> states_to_batch;
                        states_to_batch.reserve(batch.size());
                        for (const auto& req : batch) {
                                states_to_batch.push_back(req.state);
                        }

                        torch::Tensor input_tensor = torch::stack(states_to_batch).to(device);

                        auto outputs = module.forward({input_tensor}).toTuple();
                        at::Tensor p_batch = outputs->elements()[0].toTensor().softmax(1).cpu();
                        at::Tensor v_batch = outputs->elements()[1].toTensor().cpu();
                        

                        for (size_t i = 0; i < batch.size(); ++i) {
                                batch[i].promise.set_value({p_batch.index({(int)i}), v_batch[i][0].item<float>()});
                        }
                        
                } catch (const std::exception& e) {
                        std::cerr << "Inference failed!" << std::endl;
                            for (auto& req : batch) {
                                req.promise.set_value({torch::zeros({13932}), 0.0f}); 
                            }
                } catch (...) {
             std::cerr << "Unknown error in batcher_loop" << std::endl;
             for (auto& req : batch) {
                try {
                    req.promise.set_value({torch::zeros({13932}), 0.0f}); 
                } catch (...) {}
             }
            }
        }
}
