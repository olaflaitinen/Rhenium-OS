import os

ROOT_DIR = r"c:\Users\yunus\rhenium-os"
README_PATH = os.path.join(ROOT_DIR, "README.md")

SLOGAN = "Skolyn: Early. Accurate. Trusted."
LAST_UPDATED = "Last Updated: "
SPDX = "SPDX-License-Identifier"

def clean_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        return  # Skip binary files

    new_lines = []
    changed = False
    
    filename = os.path.basename(filepath)
    is_readme = (filepath == README_PATH)

    for line in lines:
        # 1. Remove Slogan (skip if README)
        if not is_readme and SLOGAN in line:
            # If line is just the slogan, skip it
            # If slogan is part of a sentence, we might need to be careful, but prompt says "sil" (delete)
            # Usually it's strictly "Skolyn: Early. Accurate. Trusted."
            # Let's replace it with empty string if it's standalone or specific format
            line = line.replace(SLOGAN, "")
            changed = True
            # If line is now empty or just whitespace, don't add it?
            if not line.strip():
                continue

        # 2. Remove Last Updated
        if LAST_UPDATED in line:
            continue # Skip entire line
            changed = True

        # 3. Remove SPDX
        if SPDX in line:
            continue # Skip entire line
            changed = True
            
        new_lines.append(line)

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"Cleaned {filepath}")

for root, dirs, files in os.walk(ROOT_DIR):
    if ".git" in dirs:
        dirs.remove(".git")
    if "__pycache__" in dirs:
        dirs.remove("__pycache__")
        
    for file in files:
        if file.endswith(".py") or file.endswith(".md") or file.endswith(".yaml"):
            clean_file(os.path.join(root, file))
