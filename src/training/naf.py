from src.training.amazing_utils import get_device
from src.data.dataloader import Document2Graph
from src.paths import NAF


def link_and_rec(args):
    device = get_device(args.gpu)
    test = Document2Graph('TEST-Naf', NAF / 'simple/test', device)
    test.get_info()
    return

def train_naf(args):

    if args.task == 'elab' or args.task == 'elin':
        link_and_rec(args)

    elif args.task == 'wgrp':
        raise Exception("Word Grouping not defined for NAF.")

    else:
        raise Exception("Task selected does not exists. Enter:\
            - 'elab': entity labeling\
            - 'elin': entity linking\
            - 'wgrp': word grouping")

    return