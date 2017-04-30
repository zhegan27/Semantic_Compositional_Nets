"""
Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zg27@duke.edu, July, 12, 2016

Computes the BLEU, ROUGE, METEOR, and CIDER
using the COCO metrics scripts
"""

# this requires the coco-caption package, https://github.com/tylin/coco-caption
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import json
from collections import defaultdict

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

if __name__ == '__main__':
    
    # this is the generated captions
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(open('./coco_scn_5k_test.txt', 'rb') )}  
    
    # this is the ground truth captions
    dataset = json.load(open('./data/coco/dataset.json', 'r'))
    split = defaultdict(list)
    for img in dataset['images']:
        split[img['split']].append(img)
    del dataset
    
    refs = []
    for img in split['test']:
        references = [' '.join(tmp['tokens']) for tmp in img['sentences']]
        refs.append(references)
    del split
    
    refs = {idx: ref for (idx, ref) in enumerate(refs)}
    
    print score(refs, hypo)
    
    
    
    
        

