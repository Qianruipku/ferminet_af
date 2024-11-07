from ferminet import base_config
from ferminet.utils import system
from absl import logging
from ferminet import train
from ferminet.constants import MUON_MASS, PROTON_MASS

import jax.numpy as jnp
import sys

def get_config():
  # Get default options.
  cfg = base_config.default()

  # Set up molecule
  cfg.system.particles = (2, 2)
  cfg.system.charges = (-1., -1.)
  cfg.system.masses = (1., 1.)
  cfg.system.molecule = [system.Atom('X', (0, 0, 0))]
  
  # Periodic boundary conditions
  cfg.system.pbc.apply_pbc = True
  cfg.system.pbc.lattice_vectors = jnp.eye(3)
  cfg.system.pbc.convergence_radius = 5
  cfg.system.pbc.min_kpoints = None

  # Set training hyperparameters
  cfg.batch_size = 4096
  cfg.pretrain.iterations = 0
  cfg.optim.iterations = 1000 + 100

  cfg.optim.laplacian = "folx"

  cfg.log.save_frequency = 5000
  cfg.log.save_path = "checkpoint_training"
  cfg.log.restore_path = "checkpoint_training"

  # cfg.optim.optimizer = "none"

  return cfg

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)
cfg = get_config()
train.train(cfg)
