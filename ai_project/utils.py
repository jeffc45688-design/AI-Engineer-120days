from collections import Counter
def print_distribution(preds,labels):
    print("preds distribution",Counter(preds))
    print("labels distribution",Counter(labels))