# configs/minatar_rainbow.py

ENV_TYPE = "minatar"

NAME = "no-double"

NOOP_MAX = 5

DOUBLE_DQN = False
N_STEP = 3
PRIORITIZED_REPLAY = True

# ("dqn", "dueling", "distributional", "distributional_dueling")
NETWORK_TYPE = "distributional_dueling"
NOISY = True

NUM_ATOMS = 51
V_MIN = -100.0
V_MAX = 100.0

ENVIRONMENTS = [
    "MinAtar/Asterix-v1",
    "MinAtar/Breakout-v1",
    "MinAtar/Freeway-v1",
    "MinAtar/Seaquest-v1",
    "MinAtar/SpaceInvaders-v1",
]

NUM_FRAMES = 10_000_000 # Agent steps #TODO adjust

REPLAY_PERIOD = 4 # Agent steps before update
REPLAY_MEMORY_SIZE = 100_000
REPLAY_MEMORY_INIT_SIZE = 1_000

PER_ALPHA = 0.5
PER_BETA_START = 0.4
PER_BETA_FRAMES = NUM_FRAMES
PER_EPS = 1e-6

DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32
TARGET_UPDATE_EVERY = 1_000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 250_000

EVAL_EVERY = 500_000
EVAL_LEN = 125_000

ADAM_LR = 2.5e-4
ADAM_EPS = 3.125e-4