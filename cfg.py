# ===============================================
# Configer of mixed model
# Written by Ji Hongchen
# 20191220
# ==============================================


class CFG():
    def __init__(self):
        # samples in each bach
        self.BATCH_SIZE = 100
        # initial weights of som net
        self.WEIGHTSHAPE = (200, 8)
        # extend num:the bases before or after
        self.EXTEND_NUM = 49

        # LSTM input size (dim)
        self.LSTMINPUTSIZE = 8
        # LSTM hidden size
        self.LSTMHIDSIZE = 64
        # LSTM output size
        self.LSTMOUTPUTSIZE = 8
        # LSTM num of layers
        self.LSTM_NUM_LAYERS = 2
        # lstm learning rate
        self.LSTMLR = 0.001

        # som learning rate
        self.SOMTARLR = 0.005
        # som sigma
        self.SIGMA = 0.4
        # the threshold value decide push or pull
        self.PUSHTHRESHOLD = 0.05

        # lstm iters per train
        self.LSTMITER = 2
        # total iter per bitch
        self.BATCH_ITER = 2

        # how many iters to print loss
        self.LOSS_PRINT = 2
        # how many exps save in per file
        self.EXP_SAVE_TIME = 2

        # the path of whole gene sequence
        self.WHOLE_SEQ_PATH = r'/home/ji/Documents/whole_sequence/hg38.fa'
        # the path of mut_infomation (saved by csv, 5000/file)
        self.MUT_INFO_PATH = r'/home/ji/Documents/test_data/exp20200816/mut_info_20200212/'
        # the saved lstm model and som weight path
        self.MODEL_PATH = r'/home/ji/Documents/test_data/exp20200816/lstmsom_model/'
        # the path of logs
        self.LOG_PATH = r'/home/ji/Documents/test_data/exp20200816/lstmsom_log/'
        # the path of saved pictures
        self.FIG_PATH = r'/home/ji/Documents/test_data/exp20200816/fig/'
        # predict add index
        self.PRED_INDEX = ['pred_1', 'pred_2', 'pred_3', 'pred_4',
                           'pred_5', 'pred_6', 'pred_7', 'pred_8']
