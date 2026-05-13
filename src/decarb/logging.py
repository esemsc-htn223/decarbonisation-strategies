__all__ = ['logger']

import logging
import os

logger = logging.getLogger('analysis')
logger.setLevel(logging.DEBUG)

logger_fh = logging.FileHandler(os.path.join('out', 'analysis.log'))
logger_fh.setLevel(logging.DEBUG)
logger_fh.set_name('analysis_debug')
logger_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(logger_fh)