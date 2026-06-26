import json
nb = json.load(open("C:\\Users\\ROWTECH\\Desktop\\ReSearch\\ai-image-detection-research\\model\\train_mfft.ipynb", encoding="utf-8"))
c2 = "".join(nb["cells"][2]["source"])
c7 = "".join(nb["cells"][7]["source"])
for line in c2.split("\n"):
    if "epochs" in line.lower() and "cfg" in line:
        print(f"Cell 2: {line.strip()}")
for line in c7.split("\n"):
    if "NUM_EPOCHS" in line:
        print(f"Cell 7: {line.strip()}")
