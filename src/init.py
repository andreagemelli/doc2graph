from src.utils import project_tree
from src.data.download import get_data

def init():
    project_tree()
    get_data()
    print("Initialization completed!")

if __name__ == '__main__':
    init()