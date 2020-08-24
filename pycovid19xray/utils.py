import os
import sys
import logging

def set_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

DONE = False
def configure_logging():
    global DONE
    if not DONE:
        # Format of a log message.
        formatter = logging.Formatter('%(asctime)s %(levelname)s\t%(message)s')
        root_log = logging.getLogger()

        # Configuring logging to stdout, for all INFO/WARNING/ERROR messages.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_log.addHandler(console_handler)
        root_log.setLevel(logging.INFO)
        DONE = True
