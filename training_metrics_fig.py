import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("runs/RL_PacMan/2025-08-13_23-30-45-dqn_run/metrics.csv")

# Display the first 5 rows of the DataFrame
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Display information about the DataFrame
print(df.info())

# Create a figure and axes for the plots
fig, axes = plt.subplots(4, 1, figsize=(10, 20))

# Plot epsilon vs episode
axes[0].plot(df["episode"], df["epsilon"])
axes[0].set_title("Epsilon vs. Episode")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Epsilon")
axes[0].grid(True)

# Plot reward vs episode
axes[1].plot(df["episode"], df["reward"])
axes[1].set_title("Reward vs. Episode")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Reward")
axes[1].grid(True)

# Plot loss vs episode
axes[2].plot(df["episode"], df["loss"])
axes[2].set_title("Loss vs. Episode")
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Loss")
axes[2].grid(True)

# Plot moving average reward vs episode
axes[3].plot(df["episode"], df["ma_reward_20"])
axes[3].set_title("Moving Average Reward (20 episodes) vs. Episode")
axes[3].set_xlabel("Episode")
axes[3].set_ylabel("Moving Average Reward")
axes[3].grid(True)

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout()

# Save the plot to a file
fig.savefig("dqn_metrics_plots.png")
