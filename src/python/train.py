import torch
from tqdm import tqdm
import concurrent.futures
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import mcts_cpp  # Your compiled C++ module
import os

torch.set_num_threads(1) 
torch.set_num_interop_threads(1)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class ShogiNet(nn.Module):
    def __init__(self, num_blocks=10, channels=256):
        super().__init__()
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(48, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            BasicBlock(channels, channels) for _ in range(num_blocks)
        ])
        
        # Policy Head
        self.p_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 9 * 9, 13932)
        
        # Value Head
        self.v_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(9 * 9, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Tower
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p) 
        
        # Value Head
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        
        return p, v

# Self-Play 
def run_self_play(nnet):
    pool = mcts_cpp.NodePool() 

    mcts = mcts_cpp.JAMES_trainer(pool,1)
    mcts.start_new_game()
    
    state = mcts_cpp.GameState.initial() 
    
    game_history = [] 
    
    move_count = 1
    while not state.is_terminal():
        search_result = mcts.search(state, nnet, ITERATIONS)
        move_count += 1

        if str(search_result.best_move) == "None" or len(search_result.visit_counts) == 0:
            print("No valid move found (Resign/Mate). Ending game.")
            break
        
        policy_target = np.zeros(13932, dtype=np.float32)
        total_visits = sum(cnt for _, cnt in search_result.visit_counts)

        
        for move_idx, count in search_result.visit_counts:
            if move_idx < 13932:
                policy_target[move_idx] = count / total_visits
        
        encoded = nnet.get_encoded_state(state) 
        game_history.append([encoded, policy_target])
        
        best_move = search_result.best_move
        state.do_move(best_move)
        mcts.update_root(state, best_move)

        if move_count > MAX_MOVES:
            break
        

    if move_count > MAX_MOVES:
        final_result = 0.0
    else:
        final_result = state.result() 
    
    processed_samples = []
    current_reward = final_result 
    
    for i in reversed(range(len(game_history))):
        encoded, policy = game_history[i]
        current_reward = -current_reward 
        processed_samples.append((encoded, policy, current_reward))
        
    return processed_samples


#  Configuration
MODEL_PATH = "shogi_net.pt"

ITERATIONS = 800  # MCTS simulations per move
GAMES_PER_EPOCH = 500
MAX_MOVES = 250

BATCH_SIZE = 128
QUEUE_SIZE = 128
NUM_WORKERS = 256

LEARNING_RATE = 0.01

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShogiNet().to(device)
    
    script_model = torch.jit.script(model)
    script_model.save(MODEL_PATH)
    
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(100):
        print(f"Epoch {epoch+1}...")
        
        cpp_net = mcts_cpp.NeuralNetwork(MODEL_PATH, QUEUE_SIZE)
        dataset = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_game = [executor.submit(run_self_play, cpp_net) for _ in range(GAMES_PER_EPOCH)]
            
            with tqdm(total=GAMES_PER_EPOCH, desc=f"Epoch {epoch+1} Self-Play", unit="game") as pbar:
                for future in concurrent.futures.as_completed(future_to_game):
                    try:
                        samples = future.result()
                        dataset.extend(samples)
                    except Exception as exc:
                        pbar.write(f"  Game generated an exception: {exc}")
                    finally:
                        pbar.update(1)

            
        states = torch.tensor(np.array([s[0] for s in dataset])).float().to(device)
        pis = torch.tensor(np.array([s[1] for s in dataset])).float().to(device)
        vs = torch.tensor(np.array([s[2] for s in dataset])).float().to(device)
        
        model.train()
        dataset_torch = TensorDataset(states, pis, vs)
        loader = DataLoader(dataset_torch, batch_size=BATCH_SIZE, shuffle=True)
        
        total_loss = 0
        for b_states, b_pis, b_vs in loader:
            optimizer.zero_grad()
            pred_pis_logits, pred_vs = model(b_states)
            
            # Loss =  Value MSE + Policy CrossEntropy
            loss_v = nn.MSELoss()(pred_vs.squeeze(), b_vs)
            
            # Policy Loss (Cross Entropy with Softmax included in LogSoftmax)
            log_pis = nn.LogSoftmax(dim=1)(pred_pis_logits)
            loss_p = -(b_pis * log_pis).sum(dim=1).mean()
            
            loss = loss_v + loss_p
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Loss: {total_loss / len(loader):.4f}")
        
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save(MODEL_PATH)
