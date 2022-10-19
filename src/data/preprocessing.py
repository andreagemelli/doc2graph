import torch
import torchvision
import numpy as np
from scipy.optimize import linprog
import os
from PIL import ImageDraw, Image
import json
import pytesseract
from pytesseract import Output

from src.paths import DATA, FUNSD_TEST


def scale_back(r, w, h): return [int(r[0]*w),
                                 int(r[1]*h), int(r[2]*w), int(r[3]*h)]


def center(r): return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)


def isIn(c, r):
    if c[0] < r[0] or c[0] > r[2]:
        return False
    elif c[1] < r[1] or c[1] > r[3]:
        return False
    else:
        return True


def match_pred_w_gt(bbox_preds: torch.Tensor, bbox_gts: torch.Tensor, links_pair: list):
    bbox_iou = torchvision.ops.box_iou(boxes1=bbox_preds, boxes2=bbox_gts)
    bbox_iou = bbox_iou.numpy()

    A_ub = np.zeros(shape=(
        bbox_iou.shape[0] + bbox_iou.shape[1], bbox_iou.shape[0] * bbox_iou.shape[1]))
    for r in range(bbox_iou.shape[0]):
        st = r * bbox_iou.shape[1]
        A_ub[r, st:st + bbox_iou.shape[1]] = 1
    for j in range(bbox_iou.shape[1]):
        r = j + bbox_iou.shape[0]
        A_ub[r, j::bbox_iou.shape[1]] = 1
    b_ub = np.ones(shape=A_ub.shape[0])

    assignaments_score = linprog(
        c=-bbox_iou.reshape(-1), A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="highs-ds")
    #assignaments_score = linprog(c=-bbox_iou.reshape(-1), bounds=(0, 1), method="highs-ds")
    # print(assignaments_score)
    if not assignaments_score.success:
        print("Optimization FAILED")
    assignaments_score = assignaments_score.x.reshape(bbox_iou.shape)
    assignaments_ids = assignaments_score.argmax(axis=1)

    # matched
    opt_assignaments = {}
    for idx in range(assignaments_score.shape[0]):
        if (bbox_iou[idx, assignaments_ids[idx]] > 0.5) and (assignaments_score[idx, assignaments_ids[idx]] > 0.9):
            opt_assignaments[idx] = assignaments_ids[idx]
    # unmatched predictions
    false_positive = [idx for idx in range(
        bbox_preds.shape[0]) if idx not in opt_assignaments]
    # unmatched gts
    false_negative = [idx for idx in range(
        bbox_gts.shape[0]) if idx not in opt_assignaments.values()]

    gt2pred = {v: k for k, v in opt_assignaments.items()}
    link_false_neg = []
    for link in links_pair:
        if link[0] in false_negative or link[1] in false_negative:
            link_false_neg.append(link)

    if len(links_pair) != 0:
        rate = len(link_false_neg) / len(links_pair)
    else:
        rate = 0
    return {"pred2gt": opt_assignaments, "gt2pred": gt2pred, "false_positive": false_positive, "false_negative": false_negative, "n_link_fn": int(len(link_false_neg) / 2), "link_loss": rate, "entity_loss": len(false_positive) / (len(false_positive) + len(opt_assignaments.keys()))}


def get_objects(path, mode):
    # TODO given a document, apply OCR or Yolo to detect either words or entities.
    return


def load_predictions(path_preds, path_gts, path_images, debug=False):
    # TODO read txt file and pass bounding box to the other function.

    boxs_preds = []
    boxs_gts = []
    links_gts = []
    labels_gts = []
    texts_ocr = []
    all_paths = []

    for img in os.listdir(path_images):
        all_paths.append(os.path.join(path_images, img))
        w, h = Image.open(os.path.join(path_images, img)).size
        texts = pytesseract.image_to_data(Image.open(
            os.path.join(path_images, img)), output_type=Output.DICT)
        tp = []
        n_elements = len(texts['level'])
        for t in range(n_elements):
            if int(texts['conf'][t]) > 50 and texts['text'][t] != ' ':
                b = [texts['left'][t], texts['top'][t], texts['left'][t] +
                     texts['width'][t], texts['top'][t] + texts['height'][t]]
                tp.append([b, texts['text'][t]])
        texts_ocr.append(tp)
        preds_name = img.split(".")[0] + '.txt'
        with open(os.path.join(path_preds, preds_name), 'r') as preds:
            lines = preds.readlines()
            boxs = list()
            for line in lines:
                scaled = scale_back([float(c)
                                    for c in line[:-1].split(" ")[1:]], w, h)
                sw, sh = scaled[2] / 2, scaled[3] / 2
                boxs.append([scaled[0] - sw, scaled[1] - sh,
                            scaled[0] + sw, scaled[1] + sh])
            boxs_preds.append(boxs)

        gts_name = img.split(".")[0] + '.json'
        with open(os.path.join(path_gts, gts_name), 'r') as f:
            form = json.load(f)['form']
            boxs = list()
            pair_labels = []
            ids = []
            labels = []
            for elem in form:
                boxs.append([float(e) for e in elem['box']])
                ids.append(elem['id'])
                labels.append(elem['label'])
                [pair_labels.append(pair) for pair in elem['linking']]

            for p, pair in enumerate(pair_labels):
                pair_labels[p] = [ids.index(pair[0]), ids.index(pair[1])]

            boxs_gts.append(boxs)
            links_gts.append(pair_labels)
            labels_gts.append(labels)

    all_links = []
    all_preds = []
    all_labels = []
    all_texts = []
    dropped_links = 0
    dropped_entity = 0

    for p in range(len(boxs_preds)):
        d = match_pred_w_gt(torch.tensor(
            boxs_preds[p]), torch.tensor(boxs_gts[p]), links_gts[p])
        dropped_links += d['link_loss']
        dropped_entity += d['entity_loss']
        links = list()

        for link in links_gts[p]:
            if link[0] in d['false_negative'] or link[1] in d['false_negative']:
                continue
            else:
                links.append([d['gt2pred'][link[0]], d['gt2pred'][link[1]]])
        all_links.append(links)

        preds = []
        labels = []
        texts = []
        for b, box in enumerate(boxs_preds[p]):
            if b in d['false_positive']:
                preds.append(box)
                labels.append('other')
            else:
                gt_id = d['pred2gt'][b]
                preds.append(box)
                labels.append(labels_gts[p][gt_id])

            text = ''
            for tocr in texts_ocr[p]:
                if isIn(center(tocr[0]), box):
                    text += tocr[1] + ' '

            texts.append(text)

        all_preds.append(preds)
        all_labels.append(labels)
        all_texts.append(texts)
    print(dropped_links / len(boxs_preds), dropped_entity / len(boxs_preds))

    if debug:
        # random.seed(35)
        # rand_idx = random.randint(0, len(os.listdir(path_images)))
        print(all_texts[0])
        rand_idx = 0
        img = Image.open(os.path.join(path_images, os.listdir(
            path_images)[rand_idx])).convert('RGB')
        draw = ImageDraw.Draw(img)

        rand_boxs_preds = boxs_preds[rand_idx]
        rand_boxs_gts = boxs_gts[rand_idx]

        for box in rand_boxs_gts:
            draw.rectangle(box, outline='blue', width=3)
        for box in rand_boxs_preds:
            draw.rectangle(box, outline='red', width=3)

        d = match_pred_w_gt(torch.tensor(rand_boxs_preds),
                            torch.tensor(rand_boxs_gts), links_gts[rand_idx])
        print(d)
        for idx in d['pred2gt'].keys():
            draw.rectangle(rand_boxs_preds[idx], outline='green', width=3)

        link_true_pos = list()
        link_false_neg = list()
        for link in links_gts[rand_idx]:
            if link[0] in d['false_negative'] or link[1] in d['false_negative']:
                link_false_neg.append(link)
                start = rand_boxs_gts[link[0]]
                end = rand_boxs_gts[link[1]]
                draw.line((center(start), center(end)), fill='red', width=3)
            else:
                link_true_pos.append(link)
                start = rand_boxs_preds[d['gt2pred'][link[0]]]
                end = rand_boxs_preds[d['gt2pred'][link[1]]]
                draw.line((center(start), center(end)), fill='green', width=3)

        precision = 0
        recall = 0
        for idx, gt in enumerate(boxs_gts):
            d = match_pred_w_gt(torch.tensor(
                boxs_preds[idx]), torch.tensor(gt), links_gts[rand_idx])
            bbox_true_positive = len(d["pred2gt"])
            p = bbox_true_positive / \
                (bbox_true_positive + len(d["false_positive"]))
            r = bbox_true_positive / \
                (bbox_true_positive + len(d["false_negative"]))
            # f1 += (2 * p * r) / (p + r)
            precision += p
            recall += r

        precision = precision / len(boxs_gts)
        recall = recall / len(boxs_gts)
        f1 = (2 * precision * recall) / (precision + recall)
        # print(f1, precision, recall)

        img.save('prova.png')

    return all_paths, all_preds, all_links, all_labels, all_texts


def save_results():
    # TODO output json of matching and check with visualization of images.
    return


if __name__ == "__main__":
    path_preds = DATA / 'FUNSD' / 'test_bbox'
    path_images = FUNSD_TEST / 'images'
    path_gts = FUNSD_TEST / 'adjusted_annotations'
    load_predictions(path_preds, path_gts, path_images, debug=True)
