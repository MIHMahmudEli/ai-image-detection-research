import json
path = "C:\\Users\\ROWTECH\\Desktop\\ReSearch\\ai-image-detection-research\\model\\train_mfft.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)
src = nb["cells"][7]["source"]
for i, line in enumerate(src):
    if line.startswith("NUM_EPOCHS = "):
        src[i] = "# Read epochs from config.py to avoid hardcoding reverts\n"
        src.insert(i+1, "NUM_EPOCHS = cfg.training.epochs\n")
        break
nb["cells"][7]["source"] = src
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Changed NUM_EPOCHS to read from cfg.training.epochs (currently 50)")
