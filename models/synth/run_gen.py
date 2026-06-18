import subprocess
import sys
import os

COLS = ["AGORA", "CARVEME"]
GRS  = ["50", "100"]

for col in COLS:
    for gr in GRS:
        print(f"\n{'='*60}")
        print(f"Running COL={col}  GR={gr} ")
        print(f"{'='*60}")
        
        # DDPM
        subprocess.run([sys.executable, "gen.py", "--col", col, "--gr", gr, "--ddpm"], check=True)
        # CTGAN
        subprocess.run([sys.executable, "gen.py", "--col", col, "--gr", gr, "--ctgan"], check=True)
        # TVAE
        subprocess.run([sys.executable, "gen.py", "--col", col, "--gr", gr, "--tvae"], check=True)
