import torch


class Evaluator:
    def __init__(self, all_true, all_outs):
        self.all_true = all_true
        self.all_outs = all_outs

    @torch.no_grad()
    def evaluate(self):
        precision, recall, f1 = calculate_metrics(self.all_true, self.all_outs, with_type=False)
        output_str = f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"
        return output_str, f1


def calculate_metrics(all_rel_true, all_rel_pred, with_type=True):
    flatrue = []
    for i, v in enumerate(all_rel_true):
        for j in v:
            try:
                head, tail, tp = j
            except:
                (head, tail), tp = j
            if with_type:
                flatrue.append((head, tail, tp, i))
            else:
                flatrue.append(((head[0], head[1]), (tail[0], tail[1]), tp, i))

    flapred = []
    for i, v in enumerate(all_rel_pred):
        for j in v:
            try:
                head, tail, tp = j
            except:
                (head, tail), tp = j
            if with_type:
                flapred.append((head, tail, tp, i))
            else:
                flapred.append(((head[0], head[1]), (tail[0], tail[1]), tp, i))

    TP = len(set(flatrue).intersection(set(flapred)))
    FP = len(flapred) - TP
    FN = len(flatrue) - TP

    if (TP + FP) == 0:
        prec = 0
    else:
        prec = TP / (TP + FP)

    if (TP + FN) == 0:
        rec = 0
    else:
        rec = TP / (TP + FN)

    # Note: It seems that you were using avg_pr and avg_re in the original code to calculate f1,
    # but they are not defined in the code snippet you provided.
    # Hence, I'm using 'prec' and 'rec' to calculate f1.
    if (prec + rec) == 0:
        f1 = 0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)

    return prec, rec, f1
