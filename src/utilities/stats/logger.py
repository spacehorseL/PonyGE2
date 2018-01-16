from time import localtime, strftime
from algorithm.parameters import params
import threading

class Logger():

    @classmethod
    def log(cls, msg, info=True):
        time_str = strftime("%H:%M:%S", localtime())
        gen_info = "[G({}/{}).E({}/{}).T({})] ".format(params['CURRENT_GENERATION'], params['GENERATIONS'], \
                                params['CURRENT_EVALUATION'], params['POPULATION_SIZE'], threading.current_thread().name) if info else ""
        print("{} |- [INFO] {}{}".format(time_str, gen_info, msg))
