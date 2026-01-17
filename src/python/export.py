import model
import torch
import torch.nn as nn
model = ShogiNet() 
model.eval()

dummy_input = torch.randn(1, 48, 9, 9)

traced_script_module = torch.jit.trace(model, dummy_input)
traced_script_module.save("models/model.pt")

print("Model exported as model.pt")
