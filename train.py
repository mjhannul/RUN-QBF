"""
Train model on qbf formulae over already trained models for lower quantification levels.
"""
import argparse
from tqdm import tqdm
import tensorflow as tf

from model import QBF_Model
from qcsp_utils import QCSP_Instance
from data_utils import load_qbf_formulas

tf.compat.v1.enable_eager_execution() # works only in eager mode currently...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--state_size', type=int, default=128, help='Size of the variable states in QBF_Model')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('-t', '--t_max', type=int, default=30, help='Number of iterations t_max for which QBF_Model runs on each instance')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('-c', '--train_size', type=int, nargs='+', default=None, help='Scope of training instances in the data directory')
    parser.add_argument('-m', '--model_dirs', type=str, nargs='+', help='Directories in which the models are stored')
    parser.add_argument('-d', '--data_path', help='A path to a training set of formulas in the (Q)DIMACS cnf format.')
    args = parser.parse_args()


    print('Loading qbf formulas...')
    _, formulas = load_qbf_formulas(args.data_path, scope=args.train_size)
    print('Converting formulas to QCSP instances...')
    instances = [QCSP_Instance.qbf_to_instance(f) for f in tqdm(formulas)]

    # Combine instances into batches
    train_batches = QCSP_Instance.batch_instances(instances, args.batch_size)

    # Fetch weights from model directories
    models = QBF_Model.get_models(args.model_dirs, train_batches[0], state_size=args.state_size)

    # Construct and train new network
    trainable_model, inference_models = models[0], models[1:]
    trainable_model.train(train_batches, inference_models, iterations=args.t_max, epochs=args.epochs)


if __name__ == '__main__':
    main()
