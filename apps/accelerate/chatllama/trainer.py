from chatllama.rlhf.actor import ActorTrainer
from chatllama.rlhf.config import Config
from chatllama.rlhf.reward import RewardTrainer
from chatllama.rlhf.trainer import RLTrainer
import os, sys

# Load config for training
path = "/root/nebullvm/apps/accelerate/chatllama/chatllama/rlhf/config.yaml"
config = Config(path=path)
# Reward Pre-Training
rw_trainer = RewardTrainer(config.reward)
rw_trainer.distill()
rw_trainer.train()

# Actor Pre-Training
act_trainer = ActorTrainer(config.actor)
act_trainer.train()

# RLHF Training
rlhf_trainer = RLTrainer(config.trainer)
rlhf_trainer.train()
rlhf_trainer.training_stats.plot()