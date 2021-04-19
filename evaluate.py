"""
Evaluate trained models on qbf formulae.
"""
import argparse
from tqdm import tqdm

from model import QBF_Model
from qcsp_utils import QCSP_Instance
from data_utils import load_qbf_formulas



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dirs', nargs='+', type=str, help='Directories in which the trained models are stored')
    parser.add_argument('-s', '--state_size', type=int, default=128, help='Size of the variable states in QBF_Model')
    parser.add_argument('-t', '--t_max', type=int, default=100, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, default=10, help='Attempts for each problem instance')
    parser.add_argument('-d', '--data_path', default=None, help='Path to the evaluation data. Expects a directory with graphs in dimacs format.')
    parser.add_argument('-c', '--evaluation_scope', type=int, nargs='+', default=None, help='Scope of evaluated instances in the data directory')
    args = parser.parse_args()


    print('Loading qbf formulas...')
    names, formulas = load_qbf_formulas(args.data_path, scope=args.evaluation_scope)
    print('Converting formulas to QCSP instances...')
    instances = [QCSP_Instance.qbf_to_instance(f) for f in tqdm(formulas)]

    # Fetch weights from model directories.
    models = QBF_Model.get_models(args.model_dirs, instances[0], state_size=args.state_size)

    # Construct and train new network.
    outermost_model, innermost_models = models[0], models[1:]

    _ = outermost_model.evaluate_instances(innermost_models, (instances, names), iterations=args.t_max, attempts=args.attempts)


if __name__ == '__main__':
    main()
