__all__ = ['logger', 'DATA_DIR', 'OUT_DIR']

import logging
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'out')
REPORT_TYPES = {'annual': 'an', 'sustainability-report': 'sr'}

logger = logging.getLogger('analysis')
logger.setLevel(logging.DEBUG)

logger_fh = logging.FileHandler(os.path.join(OUT_DIR, 'analysis.log'))
logger_fh.setLevel(logging.DEBUG)
logger_fh.set_name('analysis_debug')
logger_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(logger_fh)

