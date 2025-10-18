import torch
import torch.nn.functional as F
import time

#torch.set_num_threads(8)
B, K, D = 8132, 15, 192   # batch size, số vector trong batch, dimension
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"B={B}   K={K}    D={D}")
n = torch.randn(B, K, D, device=device)
c = torch.randn(B, D, device=device)
labels = torch.randint(0, 2, (B, K), device=device).float()

def method_mul_sum():
    # (n * c.unsqueeze(1)).sum(-1): dot product thủ công
    out = (n * c.unsqueeze(1)).sum(-1)   # [B, K]
    loss = F.binary_cross_entropy_with_logits(out, labels)
    return loss

def method_bmm():
    # bmm: [B, K, D] @ [B, D, 1] → [B, K, 1]
    out = torch.bmm(n, c.unsqueeze(-1)).squeeze(-1)  # [B, K]
    loss = F.binary_cross_entropy_with_logits(out, labels)
    return loss

def method_einsum():
    # einsum: gọn nhất
    out = torch.einsum("bkd,bd->bk", n, c)  # [B, K]
    loss = F.binary_cross_entropy_with_logits(out, labels)
    return loss

# Test correctness
#print("mul+sum loss:", method_mul_sum().item())
#print("bmm loss    :", method_bmm().item())
#print("einsum loss :", method_einsum().item())

# Benchmark tốc độ
def bench(fn, iters=100):
    for _ in range(10): fn()  # warmup
    torch.cuda.synchronize() if device=="cuda" else None
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize() if device=="cuda" else None
    return (time.time()-t0)/iters*1000

print("mul+sum:", bench(method_mul_sum), "ms/iter")
print("bmm    :", bench(method_bmm), "ms/iter")
print("einsum :", bench(method_einsum), "ms/iter")
