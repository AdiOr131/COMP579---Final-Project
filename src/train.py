import argparse
import numpy as np
from sys import stderr
from env import Game2048Env
from agent import RLAgent
import learners as L
from features import pattern

def info(msg: str):
    print(msg, file=stderr)
def error(msg: str):
    print(msg, file=stderr)

class StatsCollector:
    def __init__(self, unit=1000):
        self.unit = unit
        self.scores = []
        self.maxtile= []

    def update(self, n: int, b, score: int):
        self.scores.append(score)
        self.maxtile.append(max(b.at(i) for i in range(16)))
        if n % self.unit == 0:
            if len(self.scores)!=self.unit or len(self.maxtile)!=self.unit:
                error("wrong statistic size for show statistics"); exit(2)
            avg_score = sum(self.scores)/self.unit
            max_score = max(self.scores)
            info(f"{n}\tavg = {avg_score:.1f}\tmax = {max_score}")
            stat = [self.maxtile.count(t) for t in range(16)]
            t, c, coef = 1, 0, 100/self.unit
            while c < self.unit and t < 16:
                if stat[t]:
                    accu    = sum(stat[t:])
                    tile    = (1<<t)&-2
                    winrate = accu * coef
                    share   = stat[t] * coef
                    info(f"\t{tile}\t{winrate:.1f}%\t({share:.1f}%)")
                c += stat[t]; t += 1
            self.scores.clear(); self.maxtile.clear()


ALGO_MAP = {
    "pattern": lambda env, α: L.ApproxQLearning(
        feat_factory=lambda: pattern([0,1,2,3,4,5,6,7], iso=8),
        n_actions=env.num_moves,
        alpha=α,
        gamma=0.99
    ),
    "dqn":     lambda env, α: L.DQNLearner(
        n_actions=env.num_moves,
        lr=α,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=64,
        target_update=1000,
    ),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo",     choices=ALGO_MAP, default="pattern")
    p.add_argument("--episodes", type=int,   default=10000)
    p.add_argument("--alpha",    type=float, default=0.05,
                   help="α for pattern, lr for DQN")
    p.add_argument("--epsilon",  type=float, default=0.1)
    p.add_argument("--unit",     type=int,   default=1000,
                   help="statistic interval")
    p.add_argument("--ascii",    action="store_true")
    p.add_argument("--gui",      action="store_true")
    p.add_argument("--seed",     type=int,   default=None)
    p.add_argument("--load",     type=str,   default=None,
                   help="path to learner weights to load")
    p.add_argument("--save",     type=str,   default=None,
                   help="path to learner weights to save")
    args = p.parse_args()

    env     = Game2048Env(seed=args.seed,
                          ascii_render=args.ascii,
                          gui=args.gui)
    learner = ALGO_MAP[args.algo](env, args.alpha)
    if args.load:
        learner.load(args.load)

    agent   = RLAgent(env, learner, epsilon=args.epsilon)
    stats   = StatsCollector(unit=args.unit)

    for ep in range(1, args.episodes+1):
        score = agent.run_episode()
        stats.update(ep, env.b, score)

    if args.save:
        learner.save(args.save)
        info(f"Saved weights to {args.save}")

if __name__ == "__main__":
    main()