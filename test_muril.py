import time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification as M

name = "google/muril-base-cased"
tok = AutoTokenizer.from_pretrained(name)
model = M.from_pretrained(name, num_labels=2).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

texts = ["yaar ye bahut bakwas hai 😂"] * 8
x = tok(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to("cuda")
y = torch.tensor([0, 1] * 4).cuda()

for use_fp16 in (False, True):
    torch.cuda.reset_peak_memory_stats()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            loss = model(**x, labels=y).loss
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad()
    torch.cuda.synchronize()
    print(f"fp16={use_fp16} | {(time.time()-t0)/20:.3f}s/step | "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB | loss {loss.item():.3f}")