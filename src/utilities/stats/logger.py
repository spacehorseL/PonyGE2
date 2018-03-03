from time import localtime, strftime
from algorithm.parameters import params
import multiprocessing, threading

class Logger():
    files = {}

    @classmethod
    def log(cls, msg, info=True):
        time_str = strftime("%H:%M:%S", localtime())
        gen_info = "[G({}/{}).E({}/{}).T({})] ".format(params['CURRENT_GENERATION'], params['GENERATIONS'], \
                                params['CURRENT_EVALUATION'], params['POPULATION_SIZE'], threading.current_thread().name) if info else ""
        print("{} |- [INFO] {}{}".format(time_str, gen_info, msg), flush=True)

    @classmethod
    def fcreate(cls, fid, name):
        f = open(name, 'w')
        cls.files[fid] = f
        return f

    @classmethod
    def fwrite(cls, fid, msg):
        cls.files[fid].write(msg + "\n")

    @classmethod
    def fflush(cls, fid):
        cls.files[fid].flush()

    @classmethod
    def flush_all(cls):
        for fid, f in cls.files.items():
            f.flush()

    @classmethod
    def close_all(cls):
        for fid, f in cls.files.items():
            f.close()
