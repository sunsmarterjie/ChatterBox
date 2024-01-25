from refer import REFER
import numpy as np
import sys
import os.path as osp
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


data_root = './data'  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = 'refcocog'
splitBy = 'umd'
refer = REFER(data_root, dataset, splitBy)
sys.path.insert(0, './evaluation')
from refEvaluation import RefEvaluation

# Here's our example expression file
sample_expr_file = json.load(open("./modify_output/refcocog_test.json"))
sample_exprs = sample_expr_file['predictions']
print sample_exprs[0]
refEval = RefEvaluation(refer, sample_exprs)
refEval.evaluate()