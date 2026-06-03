__all__ = ['logger', 'DATA_DIR', 'OUT_DIR']

import logging
import os
from dataclasses import dataclass

# Data directories
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
COMPANIES_DIR = os.path.join(DATA_DIR, 'companies')
if not os.path.exists(COMPANIES_DIR):
    os.makedirs(COMPANIES_DIR)
TERMS_DIR = os.path.join(DATA_DIR, 'terms')
if not os.path.exists(TERMS_DIR):
    os.makedirs(TERMS_DIR)
LOG_DIR = os.path.join(DATA_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'out'))
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)




REPORT_TYPES = {'annual-report', 'sustainability-report'}
REPORT_TYPES_SHORT = {'annual-report': 'an', 'sustainability-report': 'sr'}
REPORT_TYPES_LONG = {'annual-report': 'annual financial report', 'sustainability-report': 'sustainability report'}

logger = logging.getLogger('analysis')
logger.setLevel(logging.DEBUG)

logger_fh = logging.FileHandler(os.path.join(LOG_DIR, 'analysis.log'))
logger_fh.setLevel(logging.DEBUG)
logger_fh.set_name('analysis_debug')
logger_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(logger_fh)

logger_llm = logging.getLogger('llm')
logger_llm.setLevel(logging.DEBUG)
logger_llm_fh = logging.FileHandler(os.path.join(LOG_DIR, 'llm.log'))
logger_llm_fh.setLevel(logging.DEBUG)
logger_llm_fh.set_name('llm_debug')
logger_llm_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger_llm.addHandler(logger_llm_fh)