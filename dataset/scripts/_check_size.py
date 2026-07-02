from pathlib import Path

src = Path("dataset/images/ai_altered")
total = 0
count = 0
for f in src.rglob("*"):
    if f.is_file():
        total += f.stat().st_size
        count += 1

print(f"AI-altered images: {count}")
print(f"Total size: {total / 1024 / 1024:.0f} MB")
print(f"Location: dataset/images/ai_altered/")
