#include "nn.h"
#include "encoder.h"
#include <chrono>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::string& model_path, int b_size) 
  : device(torch::kCUDA), queue_size(b_size), running(false) 
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

//               auto start_wait = std::chrono::high_resolution_clock::now(); 
                {
                        std::unique_lock<std::mutex> lock(queue_mtx);
                        cv.wait(lock, [this] { return !queue.empty() || !running; });
                        
                        if (!running && queue.empty()) break;

                        auto timeout = std::chrono::microseconds(150);

                        cv.wait_for(lock, timeout, [this] { 
                                return queue.size() >= (size_t)queue_size; 
                        });
                        
                        // Drain queue
                        while (!queue.empty() && batch.size() < (size_t)queue_size) {
                                batch.push_back(std::move(queue.front()));
                                queue.pop();
                        }
                }
//                auto end_wait = std::chrono::high_resolution_clock::now();
//auto start_inference = std::chrono::high_resolution_clock::now();
                
                
                
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
                        
//                        auto end_inference = std::chrono::high_resolution_clock::now();
//auto start_distribute = std::chrono::high_resolution_clock::now();

                        for (size_t i = 0; i < batch.size(); ++i) {
                                // Strategy: Pass a slice of the tensor. 
                                // .index({(int)i}) creates a view, no deep copy of data happens here.
                                batch[i].promise.set_value({p_batch.index({(int)i}), v_batch[i][0].item<float>()});
                        }
                        
                        auto end_distribute = std::chrono::high_resolution_clock::now();
//                        std::cout << "Wait: " << std::chrono::duration<float, std::milli>(end_wait - start_wait).count() << "ms "
//          << "Inference: " << std::chrono::duration<float, std::milli>(end_inference - start_inference).count() << "ms "
//          << "Distribute: " << std::chrono::duration<float, std::milli>(end_distribute - start_distribute).count() << "ms" << std::endl;
                } catch (const std::exception& e) {
                        std::cerr << "Inference failed!" << std::endl;
                                        // CRITICAL: Notify workers so they don't hang
                            for (auto& req : batch) {
                                // You could also use set_exception here
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
