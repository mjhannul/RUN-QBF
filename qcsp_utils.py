"""
This module defines classes for a constraint language and a quantified constraint satisfaction
problem (QCSP). QCSP objects are further defined in terms of QCSP clause and literal objects.
"""
import json
import itertools
import random
import numpy as np
from tqdm import tqdm


class ConstraintLanguage:
    """ Class to represent a fixed Constraint Language """

    def __init__(self, domain_size, relations):
        """
        :param domain_size: Size of the underlying domain
        :param relations: A dict of dicts specifying the relations of the language.
                        This also specifies a name for each relation.
                        I.E {'XOR': [[0, 1], [1, 0]], 'AND': [[1,1]]}
        """
        self.domain_size = domain_size
        self.relations = relations
        self.relation_names = list(relations.keys())

        # Compute characteristic matrices for each relation
        self.relation_matrices = {}
        for code, relation in self.relations.items():
            if len(code) == 1:
                M = np.zeros((self.domain_size), dtype=np.float32)
                idx = np.array(relation)
                M[idx[:, 0]] = 1.0
                self.relation_matrices[code] = M
            elif len(code) == 2:
                M = np.zeros((self.domain_size, self.domain_size), dtype=np.float32)
                idx = np.array(relation)
                M[idx[:, 0], idx[:, 1]] = 1.0
                self.relation_matrices[code] = M
            elif len(code) == 3:
                M = np.zeros(
                    (self.domain_size, self.domain_size, self.domain_size),
                    dtype=np.float32,
                )
                idx = np.array(relation)
                M[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
                self.relation_matrices[code] = M

    def save(self, path):
        """ Save details in a JSON file """
        with open(path, "w") as f:
            json.dump(
                {"domain_size": self.domain_size, "relations": self.relations},
                f,
                indent=4,
            )

    @staticmethod
    def load(path):
        """ Load instance from a JSON file """
        with open(path, "r") as f:
            data = json.load(f)

        language = ConstraintLanguage(data["domain_size"], data["relations"])
        return language


def or_relation(*targets):
    """
    Return an OR relation corresponding to the Boolean input values.
    :param *targets: Domain values for a QCSP
    :return: list of value sequences excluding the negated input sequence.
            E.g., or_relation(1,0) is [[1,0], [1,1], [0,0]].
    """
    relation = []
    for elements in itertools.product([0, 1], repeat=len(targets)):
        if sum(map(lambda x, y: abs(x - y), targets, elements)) != len(targets):
            relation.append(list(elements))
    return relation


sat3_language = ConstraintLanguage(
    domain_size=2,
    relations={
        "111": or_relation(1, 1, 1),
        "011": or_relation(0, 1, 1),
        "100": or_relation(1, 0, 0),
        "000": or_relation(0, 0, 0),
        "11": or_relation(1, 1),
        "01": or_relation(0, 1),
        "00": or_relation(0, 0),
        "1": or_relation(1),
        "0": or_relation(0),
    },
)


class QCSP_Literal:
    """Class to represent literals of QCSP instance"""

    def __init__(self, index, level, quantifier):
        """
        :param index: A positive or negative integer for literal index
        :param level: A positive integer quantification level
        :param quantifier: 'a' for universal and 'e' for existential quantification
        """
        self.index = index
        self.level = level
        self.quantifier = quantifier

    def __str__(self):
        return str(self.index) + str(self.quantifier) + str(self.level)


class QCSP_Clause:
    """Class to represent clauses of QCSP instance"""

    def __init__(self, *variables):
        """
        :param *variables: QCSP_Literal instances
        """
        self.length = len(variables)
        self.variables = sorted(list(variables), key=lambda x: (-x.level, x.index))
        self.max_level = self.variables[0].level
        self.max_quantifier = self.variables[0].quantifier

    def __str__(self):
        variable_indices = [variable.__str__() for variable in self.variables]
        variable_string = ",".join(variable_indices)
        return "[{0}]".format(variable_string)


class QCSP_Instance:
    """Class to represent QCSP instance"""

    def __init__(self, language, matrix, n_variables, **kwargs):
        """
        :param language: A constraint language instance
        :param matrix: A list of QCSP_Clause instance
        :param n_variables: Number of variables in the QCSP instance
        """
        self.language = language
        self.n_variables = n_variables
        self.n_clauses = len(matrix)
        self.matrix = matrix
        try:
            self.outermost_quantifier = kwargs["outermost_quantifier"]
            self.quantifier_rank = kwargs["quantifier_rank"]
        except KeyError:
            max_idx = np.argmax(np.array([clause.max_level for clause in matrix]))
            self.quantifier_rank = matrix[max_idx].max_level
            self.outermost_quantifier = matrix[max_idx].max_quantifier

    def __str__(self):
        clause_indices = [clause.__str__() for clause in self.matrix]
        clause_string = ";".join(clause_indices)
        return "[{0}]".format(clause_string)

    @staticmethod
    def qbf_to_instance(formula):
        """
        :param formula: A qbf formula represented as a dictionary.
                        formula["matrix"] is a list of clauses.
                            E.g. ((X1 or X2) and (not X2 or X3)) is [[1, 2], [-2, 3]]
                        formula["prefix"] is a list for quantification.
                            E.g. forall X1 X2 exists X3 is [["a", 1, 2], ["e", 3]]
                        formula["n_variables"] is the number of variables
                        formula["n_clauses"] is the number of clauses
        :return: A CSP instance that represents the formula
        """
        if not formula["prefix"]:
            # Add prefix if not defined. E.g. if formula uploaded from dimacs.
            variables = list(
                set(abs(literal) for clause in formula["matrix"] for literal in clause)
            )
            formula["prefix"] = [["e"] + variables]

        def transform_3clause(clause, n_variables):
            """
            Return equisatisfiable list of at most ternary clauses,
            and increased n_variables
            """
            new_clause = clause[:2]
            for i in range(2, len(clause) - 1):
                n_variables += 1
                clause_i = [n_variables] + [-n_variables] + [clause[i]]
                new_clause = new_clause + clause_i
            new_clause = new_clause + clause[-1:]
            new_clauses = [new_clause[i : i + 3] for i in range(0, len(new_clause), 3)]

            return new_clauses, n_variables

        def transform_3sat(formula):
            """ Return equisatisfiable formula where all clauses at most ternary """
            matrix = formula["matrix"]
            new_level = []
            new_matrix = []
            n_variables = formula["n_variables"]
            for clause in matrix:
                if len(clause) <= 3:
                    new_matrix.append(clause)
                else:
                    new_clauses, new_n_variables = transform_3clause(
                        clause, n_variables
                    )
                    new_matrix.extend(new_clauses)
                    new_level.extend(list(range(n_variables, new_n_variables + 1)))
                    n_variables = new_n_variables

            if formula["prefix"][-1][0] == "e":
                formula["prefix"][-1].extend(new_level)
            else:
                formula["prefix"][-1].append(new_level)

            formula["matrix"] = new_matrix
            formula["n_variables"] = n_variables

            return formula

        new_formula = transform_3sat(formula)
        outermost_quantifier = new_formula["prefix"][0][0]
        quantifier_rank = len(new_formula["prefix"])

        def transform_qcsp(formula):
            """
            Return formula matrix as a list of QCSP_Clause objects.
            This operation decreases variable indices by one.
            """
            prefix = {}
            for i, level in enumerate(reversed(formula["prefix"])):
                for variable in level[1:]:
                    prefix[variable] = {"quantifier": level[0], "level": i + 1}
                    prefix[-variable] = {"quantifier": level[0], "level": i + 1}

            new_matrix = []
            for clause in formula["matrix"]:
                new_clause = []
                for literal in clause:
                    # In qdimacs format a literal may appear in matrix but not in prefix. The
                    # corresponding variable is then existentially quantified in the outermost
                    # quantifier block.
                    try:
                        new_literal = QCSP_Literal(
                            literal,
                            prefix[literal]["level"],
                            prefix[literal]["quantifier"],
                        )
                    except KeyError:
                        if outermost_quantifier == "e":
                            new_literal = QCSP_Literal(
                                literal, quantifier_rank, outermost_quantifier
                            )
                        else:
                            outermost_quantifier = "e"
                            quantifier_rank += 1
                            new_literal = QCSP_Literal(
                                literal, quantifier_rank, outermost_quantifier
                            )
                    new_clause.append(new_literal)
                new_matrix.append(QCSP_Clause(*new_clause))

            return new_matrix

        new_matrix = transform_qcsp(new_formula)

        instance = QCSP_Instance(
            sat3_language,
            new_matrix,
            new_formula["n_variables"],
            outermost_quantifier=outermost_quantifier,
            quantifier_rank=quantifier_rank,
        )

        return instance

    @staticmethod
    def merge_instances(instances):
        """
        Static method to merge a list of instances into a single instance
        :param instances: A list of QCSP instances
        :return: A QCSP instance object that contains all given instances with shifted variables
        """
        language = instances[0].language
        n_variables = 0
        n_clauses = 0
        outermost_quantifier = None
        quantifier_rank = 0

        new_matrix = []
        for instance in instances:
            for clause in instance.matrix:
                new_variables = []
                for variable in clause.variables:
                    new_index = (
                        variable.index + n_variables
                        if variable.index > 0
                        else variable.index - n_variables
                    )
                    new_variable = QCSP_Literal(
                        new_index, variable.level, variable.quantifier
                    )
                    new_variables.append(new_variable)
                new_clause = QCSP_Clause(*new_variables)
                new_matrix.append(new_clause)
            n_variables += instance.n_variables
            n_clauses += instance.n_clauses
            if instance.quantifier_rank > quantifier_rank:
                quantifier_rank = instance.quantifier_rank
                outermost_quantifier = instance.outermost_quantifier

        return QCSP_Instance(
            language,
            new_matrix,
            n_variables,
            outermost_quantifier=outermost_quantifier,
            quantifier_rank=quantifier_rank,
        )

    @staticmethod
    def batch_instances(instances, batch_size):
        """
        Static method to organize given instances into batches
        :param instances: A list of QCSP instances
        :param batch_size: The batch size
        :return: A list of QCSP instances that each consist of 'batch_size' many merged instances
        """
        n_instances = len(instances)
        n_batches = int(np.ceil(n_instances / batch_size))
        batches = []

        print("Combining instances in batches...")
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min(start + batch_size, n_instances)
            batch_instance = QCSP_Instance.merge_instances(instances[start:end])
            batches.append(batch_instance)

        return batches

    @staticmethod
    def generate_random_pi2(
        clause_sizes=(0, 0.01, 0.66, 0.30, 0.02, 0.01),
        a_in_clause=(0.80, 0.18, 0.01, 0.05, 0.05, 0),
        n_variables=(10,90),
        n_clauses=400,
        language=sat3_language,
    ):
        """
        Static method to generate random Pi2 instances of the QBF problem
        :param clause_sizes: A probability distribution for clause sizes
        :param a_in_clause: A probability distribution for the number of universal variables in clauses
        :param n_variables: A distribution of universally and existentially quantified variables
        :param n_clauses: The number of clauses
        :param language: A Constraint Language
        :return: A random QCSP Instance with the specified parameters
        """
        variables = n_variables[0] + n_variables[1]

        clauses = []
        sizes = np.random.choice(len(clause_sizes), n_clauses, p=list(clause_sizes))
        for i in range(n_clauses):
            size = sizes[i]
            odds = list(a_in_clause)[: size + 1]
            a_odds = [odd / sum(odds) for odd in odds]
            a_number = np.random.choice(size + 1, 1, p=a_odds)[0]
            a_idxs = np.random.choice(n_variables[0], a_number)
            e_idxs = np.random.choice(
                np.arange(n_variables[0], variables), size - a_number
            )
            signs = [(-1) ** random.randint(0, 1) for i in range(size)]
            a_lits = [
                QCSP_Literal(signs[i] * a_idxs[i], 2, "a") for i in range(a_number)
            ]
            e_lits = [
                QCSP_Literal(signs[i + a_number] * e_idxs[i], 1, "e")
                for i in range(size - a_number)
            ]
            clauses.append(QCSP_Clause(*(a_lits + e_lits)))

        return QCSP_Instance(language, clauses, variables)
