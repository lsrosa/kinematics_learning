import sys
from rl_zoo3.train import train

if __name__ == "__main__":
    sys.argv = ["python",
                "--algo",
                "sac",
                "--env",
                "ReacherPolicy",
                "-n",
                "500",
                "-optimize",
                "--n-trials",
                "10",
                "--pruner",
                "median"
                ]

    train()
