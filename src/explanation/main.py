import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import GCNExplanation

def main():
    algos = ['GNNExplainer', 'CaptumExplainer', 'GraphMaskExplainer']
    model_explan = GCNExplanation(algo=algos[0]) 

    for model_num in range(1, 5):
        print(model_num)
        model_explan.perform_explanation(model_num)

if __name__ == "__main__":
    main()

