import torch.nn as nn
from trl import SFTTrainer, SFTConfig

model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125-Instruct")

class AddExpert(nn.Module):                 # plain PyTorch
    def __init__(self, old_block, d):
        super().__init__()
        self.old = old_block
        self.new_expert = nn.Sequential(nn.Linear(d, 1024), nn.SiLU(), nn.Linear(1024, d))

    def forward(self, x, *a, **kw):
        out = self.old(x, *a, **kw)          # original MoE output (may be a tuple)
        y = out[0] if isinstance(out, tuple) else out
        y = y + self.new_expert(x)
        return (y, *out[1:]) if isinstance(out, tuple) else y

layer = model.model.layers[10]
layer.mlp = AddExpert(layer.mlp, model.config.hidden_size)   # surgery

for p in model.parameters(): p.requires_grad = False          # freeze base
for p in layer.mlp.new_expert.parameters(): p.requires_grad = True

trainer = SFTTrainer(model=model, train_dataset=ds, args=SFTConfig(output_dir="out"))
trainer.train()                              # trains only new_expert