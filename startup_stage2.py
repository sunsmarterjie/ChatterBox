import os

cmd = f"deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 54906 train_stage2.py "

os.system(cmd)
