""" RUN-QBF model for quantified Boolean formula problems """
import json
import os
import tensorflow as tf
from tqdm import tqdm

from qcsp_utils import sat3_language
from input_utils import symmetry, get_codes, get_input


class MessageNetwork(tf.keras.layers.Layer):
    """ Message Network that sends messages between variable states """

    def __init__(self, out_units, activation="linear", code="NN", **kwargs):
        """
        :param out_units: Length of the message vectors. We usually use the variables state
                        size for this.
        :param activation: The activation of each layer.
        :param code: The code for constraint type. E.g. "NSS" indicates ternary constraint where
                        the last two positions are symmetric.
        """
        super().__init__(**kwargs)
        self.out_units = out_units
        self.activation = activation
        self.code = code

        # Output layer for generating both messages
        if code in ("NN", "NSS"):
            self.out_layer = tf.keras.layers.Dense(
                2 * self.out_units,
                activation=activation,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(),
            )
        elif code == "NNN":
            self.out_layer = tf.keras.layers.Dense(
                3 * self.out_units,
                activation=activation,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(),
            )
        else:
            self.out_layer = tf.keras.layers.Dense(
                self.out_units,
                activation=activation,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(),
            )

        self.out_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        """
        :param inputs: A tensor of shape (m, l*h) and type float32. m is the number of constraints
                        to which this messaging function is applied. h is the length of the input
                        vectors. l is the number of literals in constraints.
        :return: A tensor of above shape, that contain the messages send to each position of
                        each clause.
        """
        # Unary network or network that is not symmetric
        if self.code in ("NN", "S", "NNN"):
            # Call output layer and batch normalization
            y = self.out_layer(inputs)
            y = self.out_norm(y)

            return y

        # Symmetric binary network
        if self.code == "SS":
            in_left, in_right = (
                inputs[:, : self.out_units],
                inputs[:, self.out_units : 2 * self.out_units],
            )
            # Combine inputs in both directions
            in_lr = tf.concat([in_left, in_right], axis=1)
            in_rl = tf.concat([in_right, in_left], axis=1)

            # Stack combined tensors along batch axis
            y = tf.concat([in_lr, in_rl], axis=0)

            # Call output layer and batch normalization
            y = self.out_layer(y)
            y = self.out_norm(y)

            # Split tensor to obtain messages in both directions
            n_edges = tf.shape(in_right)[0]
            msg_left = y[:n_edges, :]
            msg_right = y[n_edges:, :]

            return tf.concat([msg_left, msg_right], axis=1)

        # Partially symmetric ternary network. I.e. the last two positions are commutative.
        if self.code == "NSS":
            in_left, in_middle, in_right = (
                inputs[:, : self.out_units],
                inputs[:, self.out_units : 2 * self.out_units],
                inputs[:, 2 * self.out_units : 3 * self.out_units],
            )

            # Combine inputs in both directions
            msg_one = tf.concat([in_left, in_middle, in_right], axis=1)
            msg_two = tf.concat([in_left, in_right, in_middle], axis=1)

            # Stack combined tensors along batch axis
            y = tf.concat([msg_one, msg_two], axis=0)

            # Call output layer and batch normalization
            y = self.out_layer(y)
            y = self.out_norm(y)

            # Compute messages for each position
            n_edges = tf.shape(in_right)[0]

            msg_middle = y[:n_edges, self.out_units : 2 * self.out_units]
            msg_right = y[n_edges : 2 * n_edges, self.out_units : 2 * self.out_units]

            msg_left_one = y[:n_edges, : self.out_units]
            msg_left_two = y[n_edges : 2 * n_edges, : self.out_units]

            msg_left = (msg_left_one + msg_left_two) / 2

            output = tf.concat([msg_left, msg_middle, msg_right], axis=1)

            return output

        # Fully symmetric ternary network. I.e. all three positions are commutative
        # elif: self.code == "SSS":
        in_left, in_middle, in_right = (
            inputs[:, : self.out_units],
            inputs[:, self.out_units : 2 * self.out_units],
            inputs[:, 2 * self.out_units : 3 * self.out_units],
        )

        # Combine inputs in both directions
        left_one = tf.concat([in_left, in_middle, in_right], axis=1)
        left_two = tf.concat([in_left, in_right, in_middle], axis=1)
        middle_one = tf.concat([in_middle, in_left, in_right], axis=1)
        middle_two = tf.concat([in_middle, in_right, in_left], axis=1)
        right_one = tf.concat([in_right, in_left, in_middle], axis=1)
        right_two = tf.concat([in_right, in_middle, in_left], axis=1)

        # Stack combined tensors along batch axis
        y = tf.concat(
            [left_one, left_two, middle_one, middle_two, right_one, right_two],
            axis=0,
        )

        # Call output layer and batch normalization
        y = self.out_layer(y)
        y = self.out_norm(y)

        # Compute average of messages for each position
        n_edges = tf.shape(in_right)[0]

        msg_left_one = y[:n_edges, :]
        msg_left_two = y[n_edges : 2 * n_edges, :]
        msg_middle_one = y[2 * n_edges : 3 * n_edges, :]
        msg_middle_two = y[3 * n_edges : 4 * n_edges, :]
        msg_right_one = y[4 * n_edges : 5 * n_edges, :]
        msg_right_two = y[5 * n_edges : 6 * n_edges, :]

        msg_left = (msg_left_one + msg_left_two) / 2
        msg_middle = (msg_middle_one + msg_middle_two) / 2
        msg_right = (msg_right_one + msg_right_two) / 2

        return tf.concat([msg_left, msg_middle, msg_right], axis=1)


class Run_QCSP_Cell(tf.keras.layers.Layer):
    """
    The RNN Cell used by RUN-CSP. Implements the cell of the network as specified for
    tf.keras.layers.RNN
    """

    def __init__(self, network):
        """
        :param network: The Run_QCSP instance that the cell belongs to
        """
        super().__init__()
        self.language = network.language
        self.state_size = [network.state_size, network.state_size]
        self.relations = self.language.relation_names

        self.relation_matrices = self.language.relation_matrices
        self.output_units = (
            self.language.domain_size if self.language.domain_size > 2 else 1
        )

        # Batch normalization layer to normalize recieved messages
        self.normalize = tf.keras.layers.BatchNormalization()

        # LSTM Cell to update variable states
        self.update = tf.keras.layers.LSTMCell(
            self.state_size[0],
            use_bias=True,
            bias_regularizer=tf.keras.regularizers.l2(),
            kernel_regularizer=tf.keras.regularizers.l2(),
            recurrent_regularizer=tf.keras.regularizers.l2(),
        )

        # Trainable linear reduction to map variable states to logits before softmax/sigmoid
        self.out_reduction = tf.keras.layers.Dense(
            self.output_units,
            activation="linear",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(),
        )

    def prepare_messages(self, states):
        """ Prepare messages for the LSTM cell """
        # Retrieve current clauses and the number of variables
        self.n_variables = self.inputs["n_variables"]
        self.clauses = self.current_inputs["clauses"]

        # Retrieve variable states
        var_states = states[0]

        # Scatter variable states over matrix of shape (n,s), where n is the total number
        # of variables and s is the state size
        var_states = tf.scatter_nd(
            self.bound_variables, var_states, [self.n_variables, self.state_size[0]]
        )

        variable_sum = tf.zeros(
            shape=[self.n_variables, self.state_size[0]], dtype=tf.float32
        )
        for clause_code, indices in self.clauses.items():
            # Send variable states to incident clauses
            idx = {}
            clause_states = []
            for i in range(3):
                if indices.shape[1] > i:  # was len(indices[0])
                    idx[i] = tf.reshape(indices[:, i], [-1, 1])
                    clause_i = tf.reshape(
                        tf.gather_nd(var_states, idx[i]),
                        [-1, self.state_size[0]],
                    )
                    clause_states.append(clause_i)
            clause_states = tf.concat(clause_states, axis=1)

            # Call the message network of the current relation to compute messages
            messages = self.message_networks[clause_code](clause_states)

            # Multiply by clause weights
            messages = messages * self.clause_weights[clause_code]

            # Add messages from each position
            for i in range(3):
                if indices.shape[1] > i:  # was len(indices[0]) > i:
                    start = i * self.state_size[0]
                    end = start + self.state_size[0]
                    variable = tf.scatter_nd(
                        idx[i],
                        messages[:, start:end],
                        shape=[self.n_variables, self.state_size[0]],
                    )
                    variable_sum = tf.math.add(variable_sum, variable)

        # Normalize variable messages with inverse weighted degrees and batch normalization
        rec = tf.math.divide_no_nan(variable_sum, self.degree_weights)
        rec = self.normalize(rec)

        # Gather variable states to matrix of shape (n_c, s), where n_c is the number of
        # bound variables and s is the state size
        messages = tf.gather_nd(rec, self.bound_variables)

        return messages

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """ Generate initial states for each variable in the given instance """
        var_states = tf.random.normal([batch_size, self.state_size[0]])
        long_states = tf.zeros([batch_size, self.state_size[1]])
        return var_states, long_states

    def call(self, x, states):
        """
        :param x: dummy input required by the LSTM cell wrapper
        :param states: The tuple with the current state tensors
        :return: A tensor with the next output for each variable and a tuple with the next states
        """
        # Prepare messages
        rec = self.prepare_messages(states)

        # Apply LSTM cell to update states
        _, (var_states, long_states) = self.update(rec, [states[0], states[1]])

        # Compute logits
        logits = self.out_reduction(var_states)

        return logits, (var_states, long_states)


class Run_QCSP_Model(tf.keras.Model):
    """
    :param model_dir: The directory to store the trained model in
    :param language: A Constraint_Language instance that specifies the underlying
                    constraint language
    :param state_size: The length of the variable state vectors
    :param codes: A tuple of clause codes, where "c" encodes variables from current level,
                and "a" and "e" universally and existentially quantified variables at levels
                below. E.g. the codes are ("c", "e") for a Pi2 model.
    :param model_dir: The directory in which the model is stored
    """

    def __init__(
        self, language, state_size=128, codes=("c", "e"), model_dir=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.language = language
        self.state_size = state_size
        self.relation_matrices = language.relation_matrices

        # Training parameters
        self.learning_rate = 0.001
        self.decay_steps = 2000
        self.decay_rate = 0.1

        # Optimization and tracking
        self.rate = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate,
            self.decay_steps,
            self.decay_rate,
            staircase=True,
            name=None,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.rate)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.conflict_tracker = tf.keras.metrics.Mean(name="conflict")

        # Instantiate RNN cell
        self.cell = Run_QCSP_Cell(self)

        # Define message networks for the RNN cell over Pi^P_2-instances
        self.message_networks = {
            code: MessageNetwork(self.state_size, code=symmetry(code), name=code)
            for code in get_codes(list(codes), domain=["0", "1"])
        }
        self.cell.message_networks = self.message_networks

        # Use keras RNN class for the recurrent neural network
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)

    def call(self, inputs, level=2):
        """
        Run_QCSP_Cell wrapped in RNN layer
        :param inputs: The dictionary of inputs obtained by get_input method
        :return: 2-D tensor of shape (n,c). n is the number of variables, c the domain size of
                the constraint language
        """
        self.inputs = inputs
        self.current_inputs = inputs["level" + str(level)]
        self.bound_variables = tf.reshape(
            self.current_inputs["bound_variables"], [-1, 1]
        )
        self.bound_n_variables = self.bound_variables.shape[0]
        self.iterations = inputs["iterations"]

        self.cell.inputs, self.cell.current_inputs = self.inputs, self.current_inputs
        self.cell.bound_variables, self.cell.language = (
            self.bound_variables,
            self.language,
        )

        # Get clause and degree weights
        self.cell.clause_weights, self.cell.degree_weights = self.get_clause_weights()

        # Define dummy input
        x = tf.zeros([self.bound_n_variables, self.iterations, 1])

        # Compute logits for bound variables
        logits = self.rnn(x)

        return logits

    def get_clause_weights(self):
        """
        Computes the weights of each clause at current level. I.e., the weight of clause C is w,
        where (1-w) is the weight of soft assignments for free variables satisfying C.
        Computes also the corresponding weighted degree of each variable at current level.
        :return: Dictionary of 2-D tensors of shape (c,1), where c is the number of clauses,
                and 2-D tensor of shape (v,1), where v is the number of variables at current level.
        """
        self.logits, self.n_variables = (
            self.inputs["logits"],
            self.inputs["n_variables"],
        )
        current_free_var_clauses, current_clauses = (
            self.current_inputs["free_var_clauses"],
            self.current_inputs["clauses"],
        )

        # Build soft assignment and its negation from logits, together with a dummy assignment
        soft_assignment = tf.nn.sigmoid(self.logits)
        soft_negated = 1 - soft_assignment
        dummy = tf.ones_like(soft_assignment)
        soft_concat = tf.concat([soft_assignment, soft_negated, dummy], axis=1)

        clause_weights = {}
        degree_weights = tf.zeros(shape=[self.n_variables], dtype=tf.float32)
        for clause_code in current_free_var_clauses.keys():

            # Gather probabilities that the free literals are false
            clauses_free = current_free_var_clauses[clause_code]
            false_free = tf.gather_nd(soft_concat, clauses_free)

            # Compute clause weights
            probabilities = tf.math.reduce_prod(false_free, axis=1, keepdims=True)
            clause_weights[clause_code] = probabilities

            # Compute degree weights
            clauses = current_clauses[clause_code]
            clause_size = clauses.shape[1]
            updates = tf.tile(probabilities, [1, clause_size])
            updates = tf.reshape(updates, [-1])
            indices = tf.reshape(clauses, [-1])
            indices = tf.expand_dims(indices, -1)
            degree_weights = tf.tensor_scatter_nd_add(degree_weights, indices, updates)

        degree_weights = tf.reshape(degree_weights, shape=[-1, 1])

        return clause_weights, degree_weights

    def build_loss(self, inputs, models, level=2, eval_only=False):
        """
        Computes the loss for training RUN-CSP
        :param inputs: The dictionary of inputs
        :param models: The models for subsequent levels
        :param level: The quantification level at which the loss is computed
        :param eval_only: Set to true if the function is called in evaluation mode
        :return: 1-D tensor that contains the loss of each iteration, weighted by a discount factor
        """
        # Retrieve list of clauses and its length
        self.all_clauses, self.n_clauses = inputs["all_clauses"], inputs["n_clauses"]

        # Update input logits with computed new logits for bound variables
        new_logits = self(inputs, level=level)
        input_logits = tf.tile(self.logits, [1, self.iterations])
        input_logits = tf.expand_dims(input_logits, -1)
        logits = tf.tensor_scatter_nd_update(
            input_logits, self.bound_variables, new_logits
        )

        # Recursively build loss (prohibitively expensive if level > 2)
        if level > 1:
            # Fetch next model in the list
            next_model = models[0]

            # Compute loss for each time step
            loss = tf.zeros((self.iterations,), dtype=tf.float32)
            for i in range(self.iterations):
                # Compute only full number of iterations in evaluation mode
                if eval_only and i != self.iterations - 1:
                    continue

                inputs["logits"] = logits[:, i, :]
                loss_timestep = next_model.build_loss(
                    inputs, models[1:], level=level - 1, eval_only=eval_only
                )

                loss = tf.tensor_scatter_nd_update(loss, [[i]], [loss_timestep])

                # Store final assignment and clause indices for evaluation
                if i == self.iterations - 1:  # not necessary
                    self.eval_phi = next_model.eval_phi
                self.idx = next_model.idx

        # Base step of recursive build loss
        else:
            if self.language.domain_size == 2:
                soft_values = tf.reshape(
                    tf.nn.sigmoid(logits), [self.n_variables, self.iterations, 1]
                )
                self.phi = tf.concat([1.0 - soft_values, soft_values], axis=2)
            else:
                self.phi = tf.nn.softmax(logits, axis=2)

            # Store final assignment for evaluation
            assignment = tf.cast(tf.argmax(self.phi, axis=2), dtype=tf.int32)
            self.eval_phi = assignment[:, -1]

            # Reshape phi to combine all iterations for each variable
            all_phi = tf.reshape(
                self.phi,
                [self.n_variables, self.iterations * self.language.domain_size],
            )

            relation_losses = []
            self.idx = {"0": {}, "1": {}, "2": {}}
            clause_phi = {}

            # Compute losses for each clause and iteration
            for code, clauses in self.all_clauses.items():
                # Order opposing predictions according to clauses
                for i in range(3):
                    if len(clauses[0]) > i:
                        self.idx[str(i)][code] = tf.reshape(clauses[:, i], [-1, 1])
                        clause_phi[i] = tf.reshape(
                            tf.gather_nd(all_phi, self.idx[str(i)][code]),
                            [-1, self.language.domain_size],
                        )

                M = self.relation_matrices[code]

                # Compute matrix product for each clause
                if len(code) == 1:
                    M = tf.reshape(M, [1, -1])
                    clause_relation_loss = tf.reduce_sum(clause_phi[0] * M, axis=1)

                if len(code) == 2:
                    clause_relation_loss = tf.reduce_sum(
                        tf.matmul(clause_phi[0], M) * clause_phi[1], axis=1
                    )

                if len(code) == 3:
                    clause_first_loss = tf.reduce_sum(
                        tf.matmul(clause_phi[1], M[0]) * clause_phi[2], axis=1
                    )
                    clause_first_loss = tf.reshape(clause_first_loss, [-1, 1])
                    clause_second_loss = tf.reduce_sum(
                        tf.matmul(clause_phi[1], M[1]) * clause_phi[2], axis=1
                    )
                    clause_second_loss = tf.reshape(clause_second_loss, [-1, 1])

                    clause_loss_concat = tf.concat(
                        [clause_first_loss, clause_second_loss], axis=1
                    )
                    clause_relation_loss = tf.reduce_sum(
                        clause_loss_concat * clause_phi[0], axis=1
                    )

                clause_relation_loss = tf.reshape(
                    clause_relation_loss, [-1, self.iterations]
                )

                # Prevent 'inf' values by setting a positive minimum loss
                min_loss = tf.constant([[1e-6]], dtype=tf.float32)
                relation_loss = -tf.math.log(tf.math.maximum(clause_relation_loss, min_loss))
                # Compute combined loss of clauses of the current relation
                relation_loss = tf.reduce_sum(relation_loss, axis=0)
                relation_losses.append(relation_loss)

            # Sum up losses across all relations
            loss = tf.math.add_n(relation_losses)
            loss = loss / tf.cast(self.n_clauses, tf.float32)

        # Compute and apply discount factor
        discount = tf.tile(tf.constant([0.95]), [self.iterations])
        exp = tf.cast(
            tf.range(self.iterations - 1, tf.constant(-1), tf.constant(-1)),
            dtype=tf.float32,
        )
        factor = tf.pow(discount, exp)
        loss = factor * loss

        total_loss = tf.reduce_sum(loss)

        return total_loss

    def train_step(self, inputs, models):
        rank = inputs["quantifier_rank"]

        with tf.GradientTape() as tape:
            # Compute loss
            total_loss = self.build_loss(inputs, models, level=rank)

            # Gradient ascent if outermost quantifier is universal
            if inputs["outermost_quantifier"] == "a":
                total_loss = -total_loss

        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_variables)

        # Clip by norm and remove 'None' values
        # for message networks that are inactive in current batch
        grads = [
            tf.clip_by_norm(grad, 1.0)
            if grad is not None
            else tf.zeros(shape=train_vars.shape, dtype=tf.float32)
            for grad, train_vars in zip(grads, self.trainable_variables)
        ]

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update loss
        self.loss_tracker.update_state(abs(total_loss))

        # Update conflicts
        conflict_ratio = self.predict()
        self.conflict_tracker.update_state(conflict_ratio)

        return conflict_ratio

    # @tf.function
    def train(self, train_batches, models, epochs=25, iterations=30):
        """
        Train model w.r.t. already trained models at lower levels
        :param train_batches: A list of train batches
        :param models: A list of trained models for the lower quantification levels
        :param epochs: The number of epochs the model is trained
        :param iterations: The number of iterations for the LSTM cell
        """
        train_log_dir = self.model_dir
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.epochs = epochs

        outermost_quantifier = train_batches[0].outermost_quantifier
        best_conflict_ratio = 1.0 if outermost_quantifier == "e" else 0.0
        for epoch in range(epochs):
            print("Training...")
            for train_batch in tqdm(train_batches):
                train_input = get_input(instance=train_batch, iterations=iterations)

                # Train and track loss
                self.train_step(train_input, models)

            print("Epoch {}".format(epoch + 1))
            print(
                "loss {:.3f}, conflict ratio {:.4f}".format(
                    self.loss_tracker.result(), self.conflict_tracker.result()
                )
            )

            # Write results
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", self.loss_tracker.result(), step=epoch)
                tf.summary.scalar("conflict ratio", self.conflict_tracker.result(), step=epoch)

            # If conflict ratio improved, save model
            if (
                self.conflict_tracker.result() < best_conflict_ratio
                and outermost_quantifier == "e"
            ):
                self.save_checkpoint("best")
                best_conflict_ratio = self.conflict_tracker.result()
                self.save_parameters()
            elif (
                self.conflict_tracker.result() > best_conflict_ratio
                and outermost_quantifier == "a"
            ):
                self.save_checkpoint("best")
                best_conflict_ratio = self.conflict_tracker.result()
                self.save_parameters()

            # Reset trackers at epoch end
            self.loss_tracker.reset_states()
            self.conflict_tracker.reset_states()

    def predict(self):
        """ Constructs the predictions and additional metrics """

        # Compute number of conflicting clauses for the assignment
        relation_conflicts = []
        edge_conflicts = {}
        assignment = tf.reshape(self.eval_phi, [self.n_variables, 1])

        for code, clauses in self.all_clauses.items():
            # Fetch the matrix pertinent to the clause code
            M = self.relation_matrices[code]

            # Get values of the positions of each clause of type r
            vals = []
            vals.append(tf.gather_nd(assignment, self.idx["0"][code]))
            if len(clauses[0]) >= 2:
                vals.append(tf.gather_nd(assignment, self.idx["1"][code]))
            if len(clauses[0]) >= 3:
                vals.append(tf.gather_nd(assignment, self.idx["2"][code]))
            val_clause = tf.concat(vals, axis=1)

            # Count conflicting clauses of type r
            valid = tf.gather_nd(M, val_clause)
            conflicts = 1.0 - valid

            edge_conflicts[code] = conflicts
            n_conflicts = tf.reduce_sum(tf.reshape(conflicts, [-1]))
            relation_conflicts.append(n_conflicts)

        # Sum up conflicts across all relations
        conflicts = tf.add_n(relation_conflicts)

        # Add metric for relative number of conflicting clauses
        n_clauses = tf.cast(self.n_clauses, tf.float32)
        conflict_ratio = conflicts / n_clauses

        # Add to conflict tracker
        self.conflict_tracker.update_state(conflict_ratio)

        return conflict_ratio

    def evaluate_instances(self, models, named_instances, iterations=50, attempts=10):
        """
        Compute and store conflict ratios for a list of QBF instances
        :param models: list of trained models for innermost quantification levels
        :param instances: list of QCSP instances
        :param attempts: number of attempts per each instance
        :return: dictionary of results
        """
        instances, names = named_instances
        quantifier_rank = instances[0].quantifier_rank
        results = {"iterations": iterations, "attempts": attempts, "results": {}}
        for j, instance in enumerate(instances):
            inputs = get_input(instance, iterations=iterations)
            conflict_ratios = []
            print("Instance {}/{}:".format(j + 1, len(instances)))
            for i in range(attempts):
                self.build_loss(inputs, models, level=quantifier_rank, eval_only=True)
                conflict_ratio = self.predict().numpy()
                if names:
                    print(
                        "Instance {} ({}) / Attempt {}: conflict ratio {}".format(
                            j + 1, names[j], i + 1, conflict_ratio
                        )
                    )
                else:
                    print(
                        "Instance {} / Attempt {}: conflict ratio {}".format(
                            j + 1, i + 1, conflict_ratio
                        )
                    )
                conflict_ratios.append(conflict_ratio)

            avg_ratio = str(sum(conflict_ratios) / attempts)
            min_ratio = str(min(conflict_ratios))
            max_ratio = str(max(conflict_ratios))
            conflict_ratios = [str(item) for item in conflict_ratios]
            if names:
                print(
                    "Instance {} ({}): avg conflict ratio {}".format(
                        j + 1, names[j], avg_ratio
                    )
                )
                results["results"][names[j]] = {
                    "avg": avg_ratio,
                    "all": conflict_ratios,
                    "min": min_ratio,
                    "max": max_ratio,
                }
            else:
                print("Instance {}: avg conflict ratio {}".format(j + 1, avg_ratio))
                results["results"][f"Instance {str(j+1)}"] = {
                    "avg": avg_ratio,
                    "all": conflict_ratios,
                    "min": min_ratio,
                    "max": max_ratio,
                }

            with open(os.path.join(self.model_dir, "results.json"), "w") as f:
                json.dump(results, f)

        return results

    def save_checkpoint(self, name="best"):
        """
        Save the current graph and summaries in the model directory
        :param name: Name of the checkpoint
        """
        path = os.path.join(self.model_dir, f"model_{name}.ckpt")
        self.save_weights(path)
        print("Model saved in file: %s" % path)

    def has_checkpoint(self):
        """ Check if network has some checkpoint stored in the model directory """
        return os.path.exists(os.path.join(self.model_dir, "checkpoint"))

    def load_checkpoint(self, name="best"):
        """
        Load a checkpoint from the model directory
        :param name: Name of the checkpoint
        """
        path = os.path.join(self.model_dir, f"model_{name}.ckpt")
        self.load_weights(path).expect_partial()

    def save_parameters(self):
        """ Saves the constraint language and state size in the model directory """
        parameters = {
            "state_size": self.state_size,
            "iterations": int(self.iterations.numpy()),
            "epochs": self.epochs,
        }
        with open(os.path.join(self.model_dir, "parameters.json"), "w") as f:
            json.dump(parameters, f)

    @staticmethod
    def get_models(model_dirs, instance, state_size=128):
        """
        Creates models and loads weights from model directories
        :param model_dirs: A list of model directories in quantification order. I.e. model_dirs[1]
                        corresponds to the outermost and model_dirs[-1] to the innermost quantifier
                        block.
        :param instance: An instance on which the model is called to create weights
        """
        models = []
        innermost_e = (
            instance.quantifier_rank % 2 == 0 and instance.outermost_quantifier == "a"
        ) or (
            instance.quantifier_rank % 2 == 1 and instance.outermost_quantifier == "e"
        )
        innermost = "e" if innermost_e else "a"
        for i, model_dir in enumerate(reversed(model_dirs)):
            if i > 1:
                network = QBF_Model(
                    state_size=state_size, codes=("a", "e", "c"), model_dir=model_dir
                )
            elif i == 1:
                network = QBF_Model(
                    state_size=state_size, codes=(innermost, "c"), model_dir=model_dir
                )
            else:
                network = QBF_Model(
                    state_size=state_size, codes=("c"), model_dir=model_dir
                )

            # build weights by calling the network at its quantification level
            rank = i + 1
            inputs = get_input(instance)
            network(inputs, level=rank)

            # build remaining message network weights
            for code, message_network in network.message_networks.items():
                dummy_input = tf.zeros(
                    [1, int(network.state_size * (len(code) / 2))], dtype=tf.float32
                )
                message_network(dummy_input)

            if network.has_checkpoint():
                # load weights if possible.
                network.load_checkpoint()
                print(f"Weights loaded from {model_dir} for model at level {rank}")
            elif rank == len(model_dirs):
                # the outermost level may not be trained yet
                print(
                    f"Could not find weights from {model_dir} for model at level {rank}"
                )
            else:
                # the innermost levels may not be trained, which is not desirable
                print(
                    f"WARNING. Could not find weights from {model_dir} for model at level {rank}"
                )
            models.append(network)

        # return outermost model first
        return models[::-1]


class QBF_Model(Run_QCSP_Model):
    """ A RUN-CSP instance for the QBF problem """

    def __init__(self, state_size=128, codes=("e", "c"), model_dir=None):
        super().__init__(
            sat3_language, codes=codes, state_size=state_size, model_dir=model_dir
        )
