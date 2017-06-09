#
#TBD, Can I do all imports here?  Should I?
import pandas as pd
import logging 

# Global variables for the project.
tmpdir = ''  #TBD 'tmpdir/'
all_ans_df = pd.DataFrame()
all_ques_df = pd.DataFrame()
a_fname = ''
q_fname = ''
progress_msg_factor = 1

# Configure basic logging.
# Set logging level to DEBUG, INFO, WARNING, ERROR, CRITICAL
# Set level to ERROR for normal use; set to INFO or DEBUG for development.
#
# logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

# Redirect STDERR or specify a log file:
log_file = "fl_fga.log"
log_level = logging.INFO
#TBD log_level = logging.DEBUG
logging.basicConfig(filename=log_file, level=log_level, format=' %(asctime)s - %(levelname)s - %(message)s')
# Sample log cmd: logging.debug('msg text var=' + str(var))
logger = logging.getLogger(__name__)

