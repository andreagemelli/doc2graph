import argparse

from src.models.training import train
from src.models.testing import test
from src.paths import FUNSD_TRAIN, FUNSD_TEST
from src.utils import set_preprocessing
from src.globals import set_device, DEVICE

def main():
    parser = argparse.ArgumentParser(description='Training')
    
    # features
    parser.add_argument('--add-layout', '-addL', action="store_true",
                        help="add layout features to nodes")
    parser.add_argument('--add-text', '-addT', action="store_true",
                        help="add textual embeddings to nodes")
    parser.add_argument('--add-visual', '-addV', action="store_true",
                        help="add visual features to nodes")
    parser.add_argument('--add-efeat', '-addE', action="store_true",
                        help="add edge features to graphs")
    # data
    parser.add_argument("--src-data", type=str, default='FUNSD',
                        help="which data source to use. It can be FUNSD, PAU or CUSTOM")
    parser.add_argument("--data-type", type=str, default='img',
                        help="if src-data is CUSTOM, define the data source type: img or pdf.")
    # graphs
    parser.add_argument("--edge-type", type=str, default='fully',
                        help="choose the kind of connectivity in the graph. It can be: fully or knn.")
    parser.add_argument("--node-granularity", type=str, default='gt',
                        help="choose the granularity of nodes to be used. It can be: gt (if given), ocr (words) or yolo (entities).")
    parser.add_argument("--num-polar-bins", type=int, default=8,
                        help="number of bins into which discretize the space for edge polar features. It must be a power of 2: Default 8.")

    # training
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument('--test', action="store_true",
                        help="skip training")
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help="provide a weights file relative path if testing")
            
    args = parser.parse_args()

    if args.test and args.weights == None:
        raise Exception("Main exception: Provide a weights file relative path! Or train a model first.")

    print(args)
    set_preprocessing(args)
    set_device(args.gpu)

    if args.src_data == 'FUNSD':
        
        if not args.test:
            args.weights = train(FUNSD_TRAIN)

        test(args, FUNSD_TEST)

    elif args.src_data == 'PAU':
        # TODO: under refactoring
        print('PAU branch under refactoring')

    elif args.src_data == 'CUSTOM':
        # TODO: develop custom data preprocessing
        raise Exception('Main exception: "CUSTOM" source data still under development')

    else:
        raise Exception('Main exception: source data invalid. Choose from ["FUNSD", "PAU", "CUSTOM"]')
    
    return

if __name__ == '__main__':
    main()
    