import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import concurrent.futures

import mcts_cpp 
from model import ShogiNet

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"

# --- Configuration ---
MODEL_PATH = "shogi_net.pt"
ITERATIONS = 800  
GAMES_PER_EPOCH = 500
MAX_MOVES = 250
BATCH_SIZE = 64    # Training Batch Size
QUEUE_SIZE = 128     # Inference Queue Size 
NUM_WORKERS = 64   # Self-play threads per GPU
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

def run_self_play(rank, cpp_net):
    
    
    pool = mcts_cpp.NodePool() 
    mcts = mcts_cpp.JAMES_trainer(pool, 1)
    mcts.start_new_game()
    state = mcts_cpp.GameState.initial()
    
    game_history = []
    move_count = 0

    temperature_threshold = 30

    while not state.is_terminal() and move_count < MAX_MOVES:
        with torch.no_grad():
            search_result = mcts.search(state, cpp_net, ITERATIONS)
        move_count += 1
        
        if str(search_result.best_move) == "None" or len(search_result.visit_counts) == 0:
            break
        temp = 1.0 if move_count < temperature_threshold else 0.01
                
        # apply temperature to visit Counts
        visit_counts = np.array([n for _, n in search_result.visit_counts], dtype=np.float32)

        policy_target = np.zeros(13932, dtype=np.float32)
        total_visits = sum(cnt for _, cnt in search_result.visit_counts)

        for move_idx, count in search_result.visit_counts:
            if move_idx < 13932:
                policy_target[move_idx] = count / total_visits
        
        encoded, mirrored_state = mcts_cpp.encode_state_mirror(state)

        policy_mirror = np.zeros(13932, dtype=np.float32)
        for i in range(13932):
            if policy_target[i] > 0:
                mirrored_idx = mcts_cpp.get_mirrored_move_index(i)
                policy_mirror[mirrored_idx] = policy_target[i]

        game_history.append([
            (encoded, policy_target),
            (mirrored_state, policy_mirror)
        ])
        
        state.do_move(search_result.best_move)
        mcts.update_root(state, search_result.best_move)

    # Value processing
    final_result = state.result() if move_count <= MAX_MOVES else 0.0
    processed_samples = []
    current_reward = final_result
    for i in reversed(range(len(game_history))):
        (normal_enc, normal_pol), (mirror_enc, mirror_pol) = game_history[i]
        
        current_reward = -current_reward

        processed_samples.append((normal_enc, normal_pol, current_reward))
        processed_samples.append((mirror_enc, mirror_pol, current_reward))

    torch.cuda.empty_cache()
    return processed_samples

def main_worker(rank, world_size):
    setup(rank, world_size)

    device_str = f"cuda:{rank}"

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

    cpp_net = mcts_cpp.NeuralNetwork(MODEL_PATH, device_str, QUEUE_SIZE)

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
            futures = [executor.submit(run_self_play, rank, cpp_net) for _ in range(local_games)]
            
            if rank == 0:
                iterator = tqdm(concurrent.futures.as_completed(futures), total=local_games, desc="Self-Play")
            else:
                iterator = concurrent.futures.as_completed(futures)

            for f in iterator:
                try:
                    local_dataset.extend(f.result())
                except Exception as e:
                    print(f"Rank {rank} error: {e}")

        states = torch.tensor(np.array([s[0] for s in local_dataset])).float()
        pis = torch.tensor(np.array([s[1] for s in local_dataset])).float()
        vs = torch.tensor(np.array([s[2] for s in local_dataset])).float()

        dataset = TensorDataset(states, pis, vs)
        
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        total_loss = 0
        
        for b_states, b_pis, b_vs in train_loader:
            b_states = b_states.to(device, non_blocking = true) 
            b_pis    = b_pis.to(device, non_blocking = true) 
            b_vs     = b_vs.to(device, non_blocking = true) 

            optimizer.zero_grad()
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            with autocast(device_type='cuda', dtype=dtype):
                p_logits, v_pred = model(aug_states)
                
                loss_v = nn.MSELoss()(v_pred.squeeze(), aug_vs)
                loss_p = -(aug_pis * torch.log_softmax(p_logits, dim=1)).sum(dim=1).mean()
                loss = loss_v + loss_p

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
            if (epoch + 1) % 10 == 0:
                cp_path = os.path.join(CHECKPOINT_DIR, f"shogi_epoch_{epoch+1}.pt")
                script_model.save(cp_path)
                print(f"Saved checkpoint: {cp_path}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Starting DDP...")
    
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
