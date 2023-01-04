from src.utils import project_tree
from src.data.download import get_data

if __name__ == '__main__':
    project_tree()
    get_data()
    print("Initialization completed!")