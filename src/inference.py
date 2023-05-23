import torch
from torch.nn import functional as F
import os
from PIL import Image, ImageDraw
import json

from src.data.feature_builder import FeatureBuilder
from src.data.graph_builder import GraphBuilder
from src.data.preprocessing import center
from src.models.graphs import SetModel
from src.paths import CHECKPOINTS, INFERENCE
from src.training.utils import get_device
from src.utils import create_folder

pretrain = {
    'funsd': {'node_num_classes': 4, 'edge_num_classes': 2},
    'pau': {'node_num_classes': 5, 'edge_num_classes': 2}
}

def inference(weights, paths, device=-1):
    # create a graph per each file
    print("Doc2Graph Inference:")
    device = get_device(device)

    print("-> Creating graphs ...")
    gb = GraphBuilder()
    graphs, _, _, features = gb.get_graph(paths, 'CUSTOM')

    # add embedded visual, text, layout etc. features to the graphs
    print("-> Creating features ...")
    fb = FeatureBuilder(d=device)
    chunks, _ = fb.add_features(graphs, features)

    # create the model
    print("-> Creating model ...")
    model = weights[0].split("-")[0]
    pre = weights[0].split("-")[1]
    sm = SetModel(name=model, device=device)
    info = pretrain[pre]
    model = sm.get_model(info['node_num_classes'], info['edge_num_classes'], chunks, False)
    model.load_state_dict(torch.load(CHECKPOINTS / weights[0]))
    model.eval()

    # predict on graphs
    print("Predicting:")
    with torch.no_grad():
        for num, graph in enumerate(graphs):
            _, name = os.path.split(paths[num])
            name = name.split(".")[0]
            print(f" -> {name}")

            n, e = model(graph.to(device), graph.ndata['feat'].to(device))
            _, epreds = torch.max(F.softmax(e, dim=1), dim=1)
            _, npreds = torch.max(F.softmax(n, dim=1), dim=1)

            # save results
            links = (epreds == 1).nonzero(as_tuple=True)[0].tolist()
            u, v = graph.edges()
            entities = features['boxs'][num]
            contents = features['texts'][num]

            graph_img = Image.open(paths[num]).convert('RGB')
            graph_draw = ImageDraw.Draw(graph_img)

            result = []
            for i, idx in enumerate(links):
                pair = {'key': {'text': contents[u[idx]], 'box': entities[u[idx]]}, 
                        'value': {'text': contents[v[idx]], 'box': entities[v[idx]]}}
                result.append(pair)

                key_center = center(entities[u[idx]])
                value_center = center(entities[v[idx]])
                graph_draw.line((key_center, value_center), fill='violet', width=3)
                graph_draw.ellipse([(key_center[0]-4,key_center[1]-4), (key_center[0]+4,key_center[1]+4)], fill = 'green', outline='black')
                graph_draw.ellipse([(value_center[0]-4,value_center[1]-4), (value_center[0]+4,value_center[1]+4)], fill = 'red', outline='black')

            graph_img.save(INFERENCE / f'{name}.png')
            with open(INFERENCE / f'{name}.json', "w") as outfile:
                json.dump(result, outfile)