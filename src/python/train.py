import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mcts_cpp  # Your compiled C++ module
import os

#  Configuration
MODEL_PATH = "shogi_net.pt"
ITERATIONS = 100  # MCTS simulations per move
GAMES_PER_EPOCH = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01

class ShogiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(48, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # Policy Head 
        self.p_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.p_fc = nn.Linear(32 * 9 * 9, 13932) 
        
        # Value Head
        self.v_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.v_fc1 = nn.Linear(9 * 9, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        
        # Policy
        p = torch.relu(self.p_conv(x))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p) 
        
        # Value
        v = torch.relu(self.v_conv(x))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        
        return p, v

# Self-Play 
def run_self_play(nnet, pool):
    mcts = mcts_cpp.MCTS(pool)
    mcts.start_new_game()
    
    state = mcts_cpp.GameState.initial() 
    
    game_history = [] # Stores (encoded_state, policy_target)
    
    while not state.is_terminal():
        search_result = mcts.search(state, nnet, ITERATIONS)

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
        
    final_result = state.result() 
    
    processed_samples = []
    current_reward = final_result 
    
    for i in reversed(range(len(game_history))):
        encoded, policy = game_history[i]
        current_reward = -current_reward 
        processed_samples.append((encoded, policy, current_reward))
        
    return processed_samples

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShogiNet().to(device)
    
    script_model = torch.jit.script(model)
    script_model.save(MODEL_PATH)
    
    pool = mcts_cpp.NodePool() 
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(100):
        print(f"Epoch {epoch+1}...")
        
        cpp_net = mcts_cpp.NeuralNetwork(MODEL_PATH)
        dataset = []
        
        for g in range(GAMES_PER_EPOCH):
            print(f"  Self-play game {g}...")
            pool.reset() # Reset memory before new game
            samples = run_self_play(cpp_net, pool)
            dataset.extend(samples)
            
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
