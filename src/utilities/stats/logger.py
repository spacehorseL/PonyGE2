from time import localtime, strftime
from algorithm.parameters import params
import threading
import multiprocessing

class Logger():

    @classmethod
    def log(cls, msg, info=True):
        time_str = strftime("%H:%M:%S", localtime())
        gen_info = "[G({}/{}).E({}/{}).T({})] ".format(params['CURRENT_GENERATION'], params['GENERATIONS'], \
                                params['CURRENT_EVALUATION'], params['POPULATION_SIZE'], multiprocessing.current_process().name) if info else ""
        print("{} |- [INFO] {}{}".format(time_str, gen_info, msg))
