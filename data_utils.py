"""
Utility functions for loading and writing .qdimacs/.cnf files.
"""
import os
import glob
from tqdm import tqdm


def load_qdimacs_qbf(path):
    """
    Loads a qbf formula from a file in (q)dimacs cnf format
    :param path: the path to a .qdimacs/.cnf file
    :return: The qbf formula as a dictionary.
            E.g. matrix is [[1, 2], [-2, 3]] for ((X1 or X2) and (not X2 or X3)).
            prefix is [["a", 1, 3], ["e", 2]] for for all X1 X3 exists X2.
            n_variables and n_clauses are the number of variables and clauses.
    """
    with open(path, "r") as file:
        prefix = []
        matrix = []
        for line in file:
            s = line.split()
            if not s:
                continue
            elif s[0] == "p":
                n_variables = int(s[2])
                n_clauses = int(s[3])
            elif s[0] == "a" or s[0] == "e":
                assert s[-1] == "0"
                level = [s[0]] + [int(l) for l in s[1:-1]]
                prefix.append(level)
            elif s[0] != "c":
                assert s[-1] == "0"
                clause = [int(l) for l in s[:-1]]
                matrix.append(clause)

    return {
        "prefix": prefix,
        "matrix": matrix,
        "n_variables": n_variables,
        "n_clauses": n_clauses,
    }


def write_qdimacs_cnf(instance, path):
    """
    Stores a qbf formula in the qdimacs format
    :param instance: A QBF instance
    :param path: The path to a file in which formula will be stored
    """
    n_variables = instance.n_variables
    n_clauses = instance.n_clauses
    quantifier = instance.outermost_quantifier

    with open(path, "w") as f:
        # Title line
        f.write(f"p cnf {n_variables} {n_clauses}\n")

        # Prefix
        for i in range(instance.quantifier_rank, 0, -1):
            variables = set()
            line = f"{quantifier} "
            for clause in instance.matrix:
                for literal in clause.variables:
                    if literal.level == i:
                        variables.add(abs(literal.index) + 1)
            variables = sorted(list(variables))
            for variable in variables:
                line += f"{variable} "
            line += "0\n"
            f.write(line)
            quantifier = "a" if quantifier == "e" else "e"

        # Matrix
        for clause in instance.matrix:
            idxs = [
                literal.index + 1 if int(literal.index) > 0 else literal.index - 1
                for literal in clause.variables
            ]
            idxs = sorted(idxs, key=abs)
            for idx in idxs:
                f.write(f"{idx} ")
            f.write("0\n")


def load_qbf_formulas(path, scope=None):
    """Loads qbf formulas from all .qdimacs/.cnf files found under the pattern 'path'
    :param path: Path to formula files
    :param scope: List of two integers to restrict the scope of loaded formulae
    """
    paths = glob.glob(os.path.join(path, f'**/*.{"cnf"}'), recursive=True)
    paths.extend(glob.glob(os.path.join(path, f'**/*.{"qdimacs"}'), recursive=True))

    if scope:
        formulas = [load_qdimacs_qbf(p) for p in tqdm(paths[scope[0] : scope[1]])]
        names = [os.path.basename(p) for p in paths[scope[0] : scope[1]]]
        return names, formulas

    formulas = [load_qdimacs_qbf(p) for p in tqdm(paths)]
    names = [os.path.basename(p) for p in paths]

    return names, formulas
