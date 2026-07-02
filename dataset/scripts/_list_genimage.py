"""List GenImage Google Drive folder structure"""
import gdown, json

folder_id = "1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
url = f"https://drive.google.com/drive/folders/{folder_id}"

# List folder contents
output = gdown.download_folder(url, quiet=True, skip_download=True)
print(f"Total items: {len(output)}")
folders = {}
for item in output:
    parts = item.path.replace("\\", "/").split("/")
    folder = parts[0] if len(parts) > 1 else "(root)"
    if folder not in folders:
        folders[folder] = {"id": item.gdrive_id if hasattr(item, 'gdrive_id') else item.id if hasattr(item, 'id') else '?', "files": 0}
    folders[folder]["files"] += 1

print(f"\nFolders ({len(folders)}):")
for f in sorted(folders.keys()):
    info = folders[f]
    print(f"  {f:30s} id={info['id']}  ({info['files']} files)")
