from src import trainer, predictor
from tensorflow.keras.backend import clear_session
# import time
if __name__=="__main__":
    datanames = ['inscape','dworld','hanrim','implant','chest']
    exprs = [2, 1, 0] ## 0: nothing, 1: cutout only, 2: all
    
    dataname = datanames[0]
    expr = exprs[1]
    trainer.main(dataname, expr)
    predictor.main(expr)
    clear_session()