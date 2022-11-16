import os.path as osp

DEFAULT_BACKEND = {
    'ppo': 'pytorch'
}

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching
WAIT_BEFORE_LAUNCH = 5

print(DEFAULT_DATA_DIR)