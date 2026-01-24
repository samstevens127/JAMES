import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import os
import concurrent.futures

import mcts_cpp 
from model import ShogiNet

# --- Configuration ---
MODEL_PATH = "shogi_net.pt"
ITERATIONS = 800  
GAMES_PER_EPOCH = 500
MAX_MOVES = 250
BATCH_SIZE = 128    # Training Batch Size
QUEUE_SIZE = 64     # Inference Queue Size 
NUM_WORKERS = 128   # Self-play threads per GPU
LEARNING_RATE = 0.01

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Set the device for this process
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_self_play(rank, model_path):
    device_str = f"cuda:{rank}"
    
    cpp_net = mcts_cpp.NeuralNetwork(model_path, device_str, QUEUE_SIZE)
    
    pool = mcts_cpp.NodePool() 
    mcts = mcts_cpp.JAMES_trainer(pool, 1)
    mcts.start_new_game()
    state = mcts_cpp.GameState.initial()
    
    game_history = []
    move_count = 0
    
    while not state.is_terminal() and move_count < MAX_MOVES:
        search_result = mcts.search(state, cpp_net, ITERATIONS)
        move_count += 1
        
        if str(search_result.best_move) == "None" or len(search_result.visit_counts) == 0:
            break

        policy_target = np.zeros(13932, dtype=np.float32)
        total_visits = sum(cnt for _, cnt in search_result.visit_counts)
        for move_idx, count in search_result.visit_counts:
            if move_idx < 13932:
                policy_target[move_idx] = count / total_visits
        
        encoded = cpp_net.get_encoded_state(state)
        game_history.append([encoded, policy_target])
        
        state.do_move(search_result.best_move)
        mcts.update_root(state, search_result.best_move)

    # Value processing
    final_result = state.result() if move_count <= MAX_MOVES else 0.0
    processed_samples = []
    current_reward = final_result
    for i in reversed(range(len(game_history))):
        encoded, policy = game_history[i]
        current_reward = -current_reward
        processed_samples.append((encoded, policy, current_reward))
        
    return processed_samples

def main_worker(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    model = ShogiNet().to(device)
    
    if os.path.exists(MODEL_PATH):
        loaded_script = torch.jit.load(MODEL_PATH, map_location=device)
        model.load_state_dict(loaded_script.state_dict())
        if rank == 0:
            print(f"Loaded existing model weights.")
    else:
        if rank == 0:
            print(f"No model found at {MODEL_PATH}. Initializing random weights.")
            model.eval()
            script_model = torch.jit.script(model)
            script_model.save(MODEL_PATH)
            print(f"Initial random model saved to {MODEL_PATH}")

    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    for epoch in range(100):
        if rank == 0:
            model.eval()
            script_model = torch.jit.script(model.module) 
            script_model.save(MODEL_PATH)
            print(f"Epoch {epoch+1} started. Model saved for workers.")
        
        dist.barrier()

        local_games = GAMES_PER_EPOCH // world_size
        local_dataset = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(run_self_play, rank, MODEL_PATH) for _ in range(local_games)]
            
            if rank == 0:
                iterator = tqdm(concurrent.futures.as_completed(futures), total=local_games, desc="Self-Play")
            else:
                iterator = concurrent.futures.as_completed(futures)

            for f in iterator:
                try:
                    local_dataset.extend(f.result())
                except Exception as e:
                    print(f"Rank {rank} error: {e}")

        states = torch.tensor(np.array([s[0] for s in local_dataset])).float().to(device)
        pis = torch.tensor(np.array([s[1] for s in local_dataset])).float().to(device)
        vs = torch.tensor(np.array([s[2] for s in local_dataset])).float().to(device)

        dataset = TensorDataset(states, pis, vs)
        
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        total_loss = 0
        
        for b_states, b_pis, b_vs in train_loader:
            optimizer.zero_grad()
            p_logits, v_pred = model(b_states)

            loss_v = nn.MSELoss()(v_pred.squeeze(), b_vs)
            log_pis = nn.LogSoftmax(dim=1)(p_logits)
            loss_p = -(b_pis * log_pis).sum(dim=1).mean()
            
            loss = loss_v + loss_p
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Starting DDP...")
    
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
