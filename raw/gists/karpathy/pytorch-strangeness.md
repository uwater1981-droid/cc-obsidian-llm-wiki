---
title: pytorch-strangeness
url: "https://gist.github.com/karpathy/e5d58e83d9fb6ce0827f0f66b253e6fe"
author: Andrej Karpathy
gist_id: e5d58e83d9fb6ce0827f0f66b253e6fe
slug: pytorch-strangeness
fetched_at: "2026-04-17T20:00:32+08:00"
type: gist
note: pytorch_strangeness.py — PyTorch 反直觉行为记录
---

## `pytorch_strangeness.py`

```
import torch
import torch.nn as nn

torch.manual_seed(42)
x = torch.randn(2, 768)

# matrix multiply "ignores" the second row when calculating the first row
w = torch.randn(768, 768)
z1 = x[0] @ w
z2 = (x @ w)[0]
print((z1-z2).abs().max().item()) # prints 0 (should be 0, OK)

# linear does not!
m = nn.Linear(768, 768, bias=False)
with torch.no_grad():
    m.weight.copy_(w.T)
q1 = m(x[0])
q2 = m(x)[0]
print((q1-q2).abs().max().item()) # prints ~2e-5 ( should be 0?!)

# and z1 != q1
print((z1-q1).abs().max().item()) # prints ~9e-5 (should be 0?!)

```
