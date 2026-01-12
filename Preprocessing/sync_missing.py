import os
import json
import shutil
import hashlib


def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Pfade definieren
map_file = "file_map.json"
external_drive_path = input("Pfad zur externen Festplatte (Quelle): ")
target_dir = input("Pfad zum Ziel-Ordner (lokaler PC): ")

with open(map_file, 'r') as f:
    source_map = json.load(f)

for rel_path, source_hash in source_map.items():
    local_path = os.path.join(target_dir, rel_path)
    external_path = os.path.join(external_drive_path, rel_path)

    # Prüfen, ob Datei existiert und identisch ist
    if os.path.exists(local_path):
        if get_file_hash(local_path) == source_hash:
            continue  # Datei ist bereits korrekt vorhanden

    # Datei kopieren
    print(f"Kopiere: {rel_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy2(external_path, local_path)

print("\nSynchronisation abgeschlossen!")