import logging 

# Global variables for the project.
DATADIR = 'data/'
MAX_COL_WID = 20
TMP = 'tmp/'

HEADER = '''
<html>
    <head>
        <style type="text/css">
        table{
            /* max-width: 850px; */
            width: 100%;
        }

        th, td {
            overflow: auto;  /* Use auto to get H scroll bars */
            text-align: left;
            /* max-width helps line-wrap in a cell, and most code */ 
            /* samples in cells have no H.scroll when width=700px: */ 
            max-width: 700px;  
            /* max-width: 50%; Using % breaks line-wrap inside a cell */
            width: auto;    /* auto is better than using % */
        }

        pre,img {
            padding: 0.1em 0.5em 0.3em 0.7em;
            border-left: 11px solid #ccc;
            margin: 1.7em 0 1.7em 0.3em;
            overflow: auto;  /* Use auto to get H scroll bars */
        }
        </style>
    </head>
    <body>
'''

FOOTER = '''
    </body>
</html>
'''


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
log_level = logging.DEBUG
logging.basicConfig(
        filename=log_file,
        level=log_level,
        format=' %(asctime)s - %(levelname)s - %(message)s')
# Sample log cmd: logging.debug('msg text var=' + str(var))
# Sample log cmd: logging.debug('msg text var=' + var)
logger = logging.getLogger(__name__)

