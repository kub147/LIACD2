import shutil

src = "CNN/alphazero_policy_v_best.pth"
dst = "gomoku_net.pth"

shutil.copy(src, dst)
print(f"Copied {src} to {dst}")