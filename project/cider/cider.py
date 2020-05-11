# From: https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/cider/cider.py
# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from typing import AnyStr, List

from cider_scorer import CiderScorer

class Cider:
    """
    Main Class to compute the CIDEr metric
    """
    def __init__(self, docTf, n=4, sigma=6.0):
        self.scorer = CiderScorer(docTf, n, sigma)

    def compute_score(self, gts: List[AnyStr], res: List[AnyStr]) -> float:
        # Paper: [rNLG(Z, Z∗) − rNLG(Zg, Z*)]
        # Z = sampled generated report
        # Z* = ground truth report
        # Zg = greedily generated report
        # A report contains multiple sentences ...
        # How do you calculate CIDEr for this set up?
        # Idea:
        # for each sen in generated report compute CIDEr against set of sentences of GT report
        # Avg those individual scores to get your overall CIDEr
        scores = []
        for hyp in res:
            hyp_score, _ = self.scorer.compute_score(hyp, gts)
            scores.append(hyp_score)

        return np.mean(np.array(scores))


    def method(self):
        return "CIDEr"
