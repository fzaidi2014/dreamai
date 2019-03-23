# Imports

import pdb
import torch
from torchvision.models import (alexnet,densenet121,densenet161,densenet169,densenet201,resnet18,resnet34,resnet50,
                                resnet101,vgg19_bn,vgg16_bn, inception_v3)
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision.datasets.folder import*
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
from matplotlib import patches, patheffects
from cycler import cycler
import time
import os
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from os.path import isfile, join
import shutil
from PIL import Image
from PIL import ImageDraw, ImageFont
from skimage import io
from skimage import transform as sk_transform
import cv2
from sklearn.metrics import hamming_loss
import torch.nn.functional as F
from collections import Counter
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score
from ast import literal_eval
import math
from pathlib import Path
import pathlib
import pickle
import json
import collections