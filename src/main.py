import argparse

from src.training.train import train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument('--add-embs', action="store_true",
                        help="add word embeddings")
    parser.add_argument('--add-attn', action="store_true",
                        help="add attention to GCN network")
    parser.add_argument('--config', type=str, default='base',
                        help="yaml file path")
    parser.add_argument('--task', type=str, default='elab',
                        help="Training task: 'elab', 'elin' or 'wgrp") 
    parser.add_argument('--test', action="store_true",
                        help="skip training")
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help="provide a weights file relative path")       
    args = parser.parse_args()

    if args.test and args.weights == None:
        raise "Provide a weights file relative path! Or train a model first."
    
    train(args)