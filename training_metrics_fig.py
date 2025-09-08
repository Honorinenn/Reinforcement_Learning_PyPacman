import pandas as pd
import matplotlib.pyplot as plt
import os


def get_metrics_files(folder):
    metrics_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file == "metrics.csv":
                metrics_files.append(os.path.join(root, file))
    return metrics_files


folder = "runs/RL_PacMan"
metrics_files = get_metrics_files(folder)
if not metrics_files:
    print("No metrics.csv files found.")
else:
    for metrics_path in metrics_files:
        print(f"\n--- {metrics_path} ---")
        df = pd.read_csv(metrics_path)
        print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
        print(df.info())
        # Prepare output directory
        out_dir = os.path.dirname(metrics_path)
        # Epsilon plot
        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["epsilon"])
        plt.title("Epsilon vs. Episode")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "epsilon.png"))
        plt.close()
        # Reward plot
        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["reward"])
        plt.title("Reward vs. Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "reward.png"))
        plt.close()
        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["loss"])
        plt.title("Loss vs. Episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss.png"))
        plt.close()
        # Moving Average Reward plot
        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["ma_reward_50"])
        plt.title("Moving Average Reward (20 episodes) vs. Episode")
        plt.xlabel("Episode")
        plt.ylabel("Moving Average Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ma_reward_50.png"))
        plt.close()
