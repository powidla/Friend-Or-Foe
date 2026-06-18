import subprocess, sys

COLS = ["AGORA", "CARVEME"]
GRS  = ["100", "50"]
DSS  = ["TL-I", "TL-II"]

for col in COLS:
    for gr in GRS:
        for ds in DSS:
            print(f"\n{'='*60}\nRunning COL={col}  GR={gr}  DS={ds}\n{'='*60}")
            subprocess.run([sys.executable, "ftrans.py", "--col", col, "--gr", gr, "--ds", ds], check=True)
          
