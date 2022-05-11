import argparse
from src.data.download import get_data
from src.training.funsd import train_funsd
from src.utils import project_tree, set_preprocessing

def main():
    parser = argparse.ArgumentParser(description='Training')

    # init
    parser.add_argument('--init', action="store_true",
                        help="download data and prepare folders")
    
    # preprocessing
    parser.add_argument('--add-embs', action="store_true",
                        help="add textual embeddings to graphs")
    parser.add_argument('--add-visual', action="store_true",
                        help="add visual features to graphs")
    parser.add_argument('--add-eweights', action="store_true",
                        help="add edge features to graphs")
    parser.add_argument("--src-data", type=str, default='FUNSD',
                        help="which data source to use. It can be FUNSD, NAF or CUSTOM")
    parser.add_argument("--data-type", type=str, default='img',
                        help="if src-data is CUSTOM, define the data source type: img or pdf.")
    parser.add_argument("--edge-type", type=str, default='fully',
                        help="choose the kind of connectivity in the graph. It can be: fully, knn or visibility.")

    # training
    parser.add_argument("--model", type=str, default='gcn',
                        help="which model to use, which yaml file to load")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument('--task', type=str, default='elab',
                        help="Training task: 'elab', 'elin' or 'wgrp") 
    parser.add_argument('--test', action="store_true",
                        help="skip training")
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help="provide a weights file relative path if testing")
         
    args = parser.parse_args()

    if args.init:
        project_tree()
        get_data()
        print("Initialization completed!")

    else:
        set_preprocessing(args.add_embs, args.add_visual, args.add_eweights, args.src_data, args.data_type, args.edge_type)
        if args.src_data == 'FUNSD':
            if args.test and args.weights == None:
                raise Exception("Provide a weights file relative path! Or train a model first.")
            train_funsd(args)

if __name__ == '__main__':
    main()
    