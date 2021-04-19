"""
Utility functions for obtaining model inputs from QCSP instances.
"""
from itertools import product
from collections import defaultdict
import tensorflow as tf


def symmetry(code):
    """
    Return message network code from clause code.
    Positions 'S' are pairwise symmetric while positions 'N' are not.
    """
    code = list(zip(code[: int(len(code) / 2)], code[int(len(code) / 2) :]))
    if len(code) == 3:
        if code[0] == code[1] and code[1] == code[2]:
            letters = "SSS"
        elif code[1] == code[2]:
            letters = "NSS"
        else:
            letters = "NNN"
    elif len(code) == 2:
        if code[0] == code[1]:
            letters = "SS"
        else:
            letters = "NN"
    elif len(code) == 1:
        letters = "S"
    return letters


def permute(tup):
    """ Ensure symmetry on last two positions of a ternary tuple. """
    if tup[0] == tup[1] != tup[2]:
        perm = (2, 0, 1)
    elif tup[0] != tup[1] != tup[2] and tup[0] == tup[2]:
        perm = (1, 0, 2)
    else:
        perm = (0, 1, 2)
    return perm


def level(clause, current_level):
    """
    Relativize a QCSP Clause instance to a given level of quantification.
    :param clause: A QCSP Clause instance.
    :param current_level: An integer representing current level of quantification.
    :return: A dictionary representing clause at current level of quantififaction.
    """
    above_variables = [
        variable for variable in clause.variables if variable.level > current_level
    ]
    current_variables = [
        variable for variable in clause.variables if variable.level == current_level
    ]
    below_variables = [
        variable for variable in clause.variables if variable.level < current_level
    ]

    # Prepare for clause code
    current_codes = [
        ("c", str(int(variable.index > 0))) for variable in current_variables
    ]
    below_codes = [
        (variable.quantifier, str(int(variable.index > 0)))
        for variable in below_variables
    ]
    codes = current_codes + below_codes
    bound_variables = current_variables + below_variables

    # Enforce symmetry on the last two positions of ternary tuples.
    if len(codes) == 3:
        a, b, c = permute(codes)
        codes = [codes[a], codes[b], codes[c]]
        bound_variables = [bound_variables[a], bound_variables[b], bound_variables[c]]

    # Create code string and variable list
    codes = [code[0] for code in codes] + [code[1] for code in codes]
    code = "".join(codes)

    # List the bound variables and the variables at the current quantification level.
    bound_variables = [abs(variable.index) - 1 for variable in bound_variables]
    current_variables = [abs(variable.index) - 1 for variable in current_variables]

    # List pairs (i, s), where i are the indices of the free literals, s encodes their signs.
    free_variables = [
        [abs(variable.index) - 1, int(variable.index > 0)]
        for variable in above_variables
    ]

    # Pad with dummy indices
    free_variables += [[0, 2]] * (2 - len(free_variables))

    clause_dict = {
        "size": len(bound_variables),
        "bound_variables": bound_variables,
        "current_variables": current_variables,
        "free_variables": free_variables,
        "clause_code": code,
    }

    return clause_dict


def get_codes(codes, domain=None):
    """
    Retrieve all possible clause codes.
    :param codes: A list of letters from "a", "e", "c" for universal, existential, and current variables.
    :param domain: A list for the QCSP problem domain. E.g., the problem domain for QBF is ["0", "1"].
    :return: possible clause codes
    """
    clause_codes = []
    for tup in product(
        product(codes, domain), product(codes, domain), product(codes, domain)
    ):
        tup = sorted(tup, key=lambda tup: (tup[0], tup[1]))
        a, b, c = permute(tup)
        code = tup[a][0] + tup[b][0] + tup[c][0] + tup[a][1] + tup[b][1] + tup[c][1]
        clause_codes.append(code)
    for tup in product(product(codes, domain), product(codes, domain)):
        tup = sorted(tup, key=lambda tup: (tup[0], tup[1]))
        code = tup[0][0] + tup[1][0] + tup[0][1] + tup[1][1]
        clause_codes.append(code)
    for tup in product(codes, domain):
        code = tup[0] + tup[1]
        clause_codes.append(code)
    return set(clause_codes)


def get_input(instance, iterations=25):
    """
    Transform a QCSP instance to an input dictionary for the RUN-QBF model.
    :param instance: A QCSP instance
    :param rank: Quantifier rank of the QCSP instance
    :return: A nested inputs dictionary for QBF model.
            E.g., inputs["levelk"]["bound_variables"] is the set of bound variables at level k.
            inputs["levelk"]["current_variables"] is the set of variables quantified at level k.
            inputs["levelk"]["clauses"][<code>] lists the bound variables of all clauses with
                clause code <code>.
            inputs["levelk"]["free_var_clauses"][<code>] lists the free variables of all clauses
                with clause code <code>.
    """
    running_rank = instance.quantifier_rank
    clause_inputs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    variable_inputs = defaultdict(lambda: defaultdict(set))
    all_clauses = defaultdict(list)

    for clause in instance.matrix:

        # Add clause to all_clauses without relativization
        sortedvars = sorted(clause.variables, key=lambda x: x.index)
        literals = [abs(variable.index) - 1 for variable in sortedvars]
        signs = [int(variable.index > 0) for variable in sortedvars]
        signs = "".join([str(sign) for sign in signs])
        # Change "001" to "100" to enfoce symmetry on the last two positions
        if signs == "001":
            signs = "100"
            literals = literals[::-1]
        all_clauses[signs].append(literals)

        # Relativize clause for each level
        while running_rank > 0:
            clause_dict = level(clause, running_rank)

            # Ignore clauses with no variables from current or subsequent levels
            if clause_dict["size"]:
                code = clause_dict["clause_code"]

                # Update the variables of current level
                variable_inputs[str(running_rank)]["bound_variables"].update(
                    set(clause_dict["bound_variables"])
                )
                variable_inputs[str(running_rank)]["current_variables"].update(
                    set(clause_dict["current_variables"])
                )

                # Add variables and free variables for current clause
                clause_inputs[str(running_rank)]["clauses"][code].append(
                    clause_dict["bound_variables"]
                )
                clause_inputs[str(running_rank)]["free_var_clauses"][code].append(
                    clause_dict["free_variables"]
                )

            running_rank -= 1
        running_rank = instance.quantifier_rank

    def default_to_regular(d):
        # Transform defaultdict to regular dict
        if isinstance(d, defaultdict):
            d = {key: default_to_regular(value) for key, value in d.items()}
        else:
            d = tf.constant(list(d), dtype=tf.int32)
        return d

    clause_inputs, variable_inputs, all_clauses = (
        default_to_regular(clause_inputs),
        default_to_regular(variable_inputs),
        default_to_regular(all_clauses),
    )

    # Create inputs dict by merging first variable and clause inputs
    inputs = {}
    for running_rank in range(1, instance.quantifier_rank + 1):
        try:
            inputs["level" + str(running_rank)] = {
                **clause_inputs[str(running_rank)],
                **variable_inputs[str(running_rank)],
            }
        except KeyError:
            pass

    inputs["n_variables"] = instance.n_variables
    inputs["n_clauses"] = instance.n_clauses
    inputs["logits"] = tf.zeros(shape=[instance.n_variables, 1])
    inputs["iterations"] = iterations
    inputs["all_clauses"] = all_clauses
    inputs["outermost_quantifier"] = instance.outermost_quantifier
    inputs["quantifier_rank"] = instance.quantifier_rank

    return inputs
