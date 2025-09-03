import os
import json


def get_metrics_files(folder):
    metrics_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file == "metrics.csv":
                metrics_files.append(os.path.join(root, file))
    return metrics_files


def load_csv(file_path):
    import csv

    data = []
    try:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    return data


def analyze_csv(data):
    stats = {}
    rewards = []
    steps = []
    episodes = 0
    for row in data:
        # Try to extract reward and step columns
        if "reward" in row:
            try:
                rewards.append(float(row["reward"]))
            except:
                pass
        if "steps" in row:
            try:
                steps.append(float(row["steps"]))
            except:
                pass
        episodes += 1
    if rewards:
        stats["average_reward"] = sum(rewards) / len(rewards)
        stats["max_reward"] = max(rewards)
        stats["min_reward"] = min(rewards)
    if steps:
        stats["average_steps"] = sum(steps) / len(steps)
    stats["total_episodes"] = episodes
    return stats


def explain_stats(stats):
    explanations = []
    if "average_reward" in stats:
        explanations.append(
            f"Average reward: {stats['average_reward']:.2f} - Indicates the agent's typical performance."
        )
    if "max_reward" in stats:
        explanations.append(
            f"Max reward: {stats['max_reward']:.2f} - Highest reward achieved in training."
        )
    if "min_reward" in stats:
        explanations.append(
            f"Min reward: {stats['min_reward']:.2f} - Lowest reward, showing worst-case performance."
        )
    if "total_episodes" in stats:
        explanations.append(
            f"Total episodes: {stats['total_episodes']} - Number of training runs."
        )
    if "average_steps" in stats:
        explanations.append(
            f"Average steps: {stats['average_steps']:.2f} - Average steps per episode."
        )
    return explanations


def main():
    folder = "runs/RL_PacMan"
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return

    metrics_files = get_metrics_files(folder)
    if not metrics_files:
        print("No metrics.csv files found.")
        return

    print(f"Found {len(metrics_files)} metrics.csv files in '{folder}'.\n")
    for file_path in metrics_files:
        data = load_csv(file_path)
        if data is None:
            continue
        stats = analyze_csv(data)
        explanations = explain_stats(stats)
        print(f"File: {file_path}")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        for explanation in explanations:
            print(f"  - {explanation}")
        print()


if __name__ == "__main__":
    main()
