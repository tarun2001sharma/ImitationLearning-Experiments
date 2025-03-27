
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Read the CSV files
    train_df = pd.read_csv("/Users/admin/Desktop/Sem4/DeepRL/assignment_1/policy/exp_local/2025.02.21/bc_multimodal_test6/train.csv")
    eval_df  = pd.read_csv("/Users/admin/Desktop/Sem4/DeepRL/assignment_1/policy/exp_local/2025.02.21/bc_multimodal_test6/eval.csv")

    # 2. Print columns for sanity check
    print("Columns in train.csv:", train_df.columns)
    print("Columns in eval.csv:", eval_df.columns)

    # Expected columns:
    # train.csv -> "step", "actor_loss", ...
    # eval.csv  -> "step", "episode_reward", ...

    # 3. Create subplots: one for training, one for evaluation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # 4. Plot training (actor_loss vs step)
    ax1.plot(train_df["step"], train_df["actor_loss"], label="Actor Loss", color="blue")
    ax1.set_ylabel("Actor Loss")
    ax1.set_title("Training Curve")
    ax1.legend()
    ax1.grid(True)

    # 5. Plot evaluation (episode_reward vs step)
    ax2.plot(eval_df["step"], eval_df["episode_reward"], label="Episode Reward", color="orange")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Episode Reward")
    ax2.set_title("Evaluation Curve")
    ax2.legend()
    ax2.grid(True)

    # 6. Adjust layout
    plt.tight_layout()

    # 7. Save the figure in the same directory as this script
    plt.savefig("train_eval_plot.png", dpi=150)

    # 8. Show the figure
    plt.show()

if __name__ == "__main__":
    main()