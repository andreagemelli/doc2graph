import argparse

from src.data.download import get_data
from src.training.funsd import train_funsd
from src.utils import project_tree, set_preprocessing
from src.training.pau import train_pau

def main():
    parser = argparse.ArgumentParser(description='Training')

    # init
    parser.add_argument('--init', action="store_true",
                        help="download data and prepare folders")
    
    # features
    parser.add_argument('--add-geom', '-addG', action="store_true",
                        help="add geometrical features to nodes")
    parser.add_argument('--add-embs', '-addT', action="store_true",
                        help="add textual embeddings to nodes")
    parser.add_argument('--add-hist', '-addH', action="store_true",
                        help="add histogram of contents to nodes")
    parser.add_argument('--add-visual', '-addV', action="store_true",
                        help="add visual features to nodes")
    parser.add_argument('--add-eweights', '-addE', action="store_true",
                        help="add edge features to graphs")
    # data
    parser.add_argument("--src-data", type=str, default='FUNSD',
                        help="which data source to use. It can be FUNSD, PAU or CUSTOM")
    parser.add_argument("--data-type", type=str, default='img',
                        help="if src-data is CUSTOM, define the data source type: img or pdf.")
    # graphs
    parser.add_argument("--edge-type", type=str, default='fully',
                        help="choose the kind of connectivity in the graph. It can be: fully, knn or visibility.")

    # training
    parser.add_argument("--model", type=str, default='GCN',
                        help="which model to use, which yaml file to load. GCN, EDGE, GAT")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument('--task', type=str, default='elab',
                        help="Training task: 'elab', 'elin' or 'wgrp") 
    parser.add_argument('--test', action="store_true",
                        help="skip training")
    parser.add_argument('--weights', '-w', nargs='+', type=str, default=None,
                        help="provide a weights file relative path if testing")
         
    args = parser.parse_args()
    print(args)

    if args.init:
        project_tree()
        get_data()
        print("Initialization completed!")

    else:
        set_preprocessing(args)
        if args.src_data == 'FUNSD':
            if args.test and args.weights == None:
                raise Exception("Main exception: Provide a weights file relative path! Or train a model first.")
            train_funsd(args)
        elif args.src_data == 'PAU':
            train_pau(args)
        elif args.src_data == 'CUSTOM':
            #TODO develop custom data preprocessing
            raise Exception('Main exception: "CUSTOM" source data still under development')
        else:
            raise Exception('Main exception: source data invalid. Choose from ["FUNSD", "PAU", "CUSTOM"]')
    
    return

if __name__ == '__main__':
    main()
    