import numpy as np
from numpy import ndarray
from utils import assert_dim, assert_same_shape
from losses import LOSS_FUNCTIONS
from sklearn.preprocessing import LabelBinarizer


def sigmoid(x: ndarray):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x: ndarray):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: ndarray):
    return np.tanh(x)


def dtanh(x: ndarray):
    return 1 - np.tanh(x) * np.tanh(x)


class RNNCell(object):
    def __init__(self):
        pass

    def forward(self, input_t, hidden_params_, params):
        '''
        a_t = tanh(a_{t - 1} * W_{aa} + x_t * W_{ax} + B_{a})
        y_t = a_t * W_{ya} + B_{y}


        Parameters
        ----------
        input_t : ndarray of size = (batch_size, input_size)

        hidden_params_ : ndarray of size = (batch_size, hidden_size)

        Returns
        ----------
        output_ : ndarray of size = (batch_size, hidden_size)
        hidden_params_out_activated : ndarray of size = (batch_size, hidden_size)
        '''
        assert_dim(input_t, 2)
        assert_dim(hidden_params_, 2)
        self.input_ = input_t
        self.hidden_params_in = hidden_params_
        self.hidden_params_out = (
            np.dot(hidden_params_, params['W_aa']['value'])
            + np.dot(input_t, params['W_ax']['value'])
            + params['b_a']['value']
        )
        # (batch_size, hidden_size)
        self.hidden_params_out_activated = np.tanh(self.hidden_params_out)
        # (batch_size, input_size)
        self.output_ = (
            np.dot(self.hidden_params_out_activated, params['W_ya']['value'])
            + params['b_y']['value']
        )
        return self.output_, self.hidden_params_out_activated

    def backward(self, output_grad, hidden_grad, params):
        '''
        a_t = tanh(a_{t - 1} * W_{aa} + x_t * W_{ax} + B_{a})
        y_t = a_t * W_{ya} + B_{y}

        Parameters
        ----------
        output_grad : ndarray of size = (batch_size, hidden_size)
            dL/dy_t

        hidden_grad : ndarray of size = (batch_size, hidden_size)
            dL/da_t

        Returns
        ----------
        input_grad : ndarray of size = (batch_size, input_size)
        hidden_input_grad : ndarray of size = (batch_size, hidden_size)
        '''
        assert_same_shape(output_grad, self.output_)
        assert_same_shape(hidden_grad, self.hidden_params_out_activated)

        # First update output parameters

        # dL/dy_t * (dy_t/d_{b_y}) (1, hidden_size)
        params['b_y']['deriv'] += output_grad.sum(axis=0)

        # dL/dy_t * (dy_t/d_{W_ya}) (hidden_size, hidden_size)
        params['W_ya']['deriv'] += np.dot(
            self.hidden_params_out_activated.T, output_grad
        )

        # Now update hidden parameters

        # dL/dy_t * (dy_t/d_{a_t}) (batch_size, hidden_size)
        d_hidden = np.dot(output_grad, params['W_ya']['value'].T)
        # dL/d_{a_t} (from parameters) (batch_size, hidden_size)
        d_hidden += hidden_grad
        # Derivative w.r.t tanh (batch_size, hidden_size)
        d_hidden = d_hidden * (1 - np.power(self.hidden_params_out_activated, 2))

        # (1, hidden_size)
        params['b_a']['deriv'] += d_hidden.sum(axis=0)
        # (hidden_size, hidden_size)
        params['W_aa']['deriv'] += np.dot(self.hidden_params_in.T, d_hidden)
        params['W_ax']['deriv'] += np.dot(self.input_.T, d_hidden)

        input_grad = np.dot(d_hidden, params['W_ax']['value'].T)
        hidden_input_grad = np.dot(d_hidden, params['W_aa']['value'].T)
        assert_same_shape(input_grad, self.input_)
        assert_same_shape(hidden_input_grad, self.hidden_params_in)
        return input_grad, hidden_input_grad


class RNN(object):
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.first = True
        self._init_parameters()

    def _init_parameters(self):
        '''
        Initialize learning parameters
        '''
        self.params = dict()

        # Get hidden params
        self.params['W_aa'] = dict()  # get a from a
        self.params['W_ax'] = dict()  # get a from x
        self.params['b_a'] = dict()  # bias

        # Get output params
        self.params['W_ya'] = dict()  # get y from a
        self.params['b_y'] = dict()  # bias

        self.params['W_aa']['value'] = np.random.normal(
            size=(self.hidden_size, self.hidden_size)
        )
        self.params['W_ax']['value'] = np.random.normal(
            size=(self.input_size, self.hidden_size)
        )
        self.params['W_ya']['value'] = np.random.normal(
            size=(self.hidden_size, self.hidden_size)
        )

        self.params['b_a']['value'] = np.random.normal(size=(1, self.hidden_size))
        self.params['b_y']['value'] = np.random.normal(size=(1, self.hidden_size))

        for key in self.params:
            self.params[key]['deriv'] = np.zeros_like(self.params[key]['value'])

    def forward(self, input_: ndarray, h0: ndarray):
        '''
        Forwards the input and initial hidden state
        Parameters
        ----------
        input_ : ndarray of size = (batch_size, sequence_length, input_size)
            containing the features of the input sequence.

        h0: ndarray of size = (batch_size, hidden_size)
            containing the initial hidden state for each element in the batch

        Returns
        ----------
        output_ : ndarray of size = (batch_size, sequence_length, hidden_size)
            containing the output features (h_t) from the last layer of the RNN, for each t

        hn : ndarray of size = (batch_size, hidden_size)
            final hidden state for each element in the batch
        '''
        assert_dim(input_, 3)
        self.input_ = input_
        (batch_size, sequence_length, input_size) = input_.shape
        assert input_size == self.input_size, (
            f"input size {input_size} does not match with initialized layer's input"
            f" size{self.input_size}"
        )

        # Initialize the layer
        if self.first:
            self.cells = [RNNCell() for _ in range(sequence_length)]
            self.first = False

        # Don't modify input hidden state!
        self.hn = np.copy(h0)

        # Initialize the outputs
        self.output_ = np.zeros((batch_size, sequence_length, self.hidden_size))
        for time_stamp in range(sequence_length):
            # (batch_size, input_size)
            input_t = np.take(input_, time_stamp, axis=1)
            self.output_[:, time_stamp, :], self.hn = self.cells[time_stamp].forward(
                input_t, self.hn, self.params
            )
        return self.output_, self.hn

    def backward(self, output_grad):
        '''
        Backpropagate the output gradients
        Parameters
        ----------
        output_grad : ndarray of size = (batch_size, sequence_length, hidden_size)

        Returns
        ----------
        input_grad : ndarray of size = (batch_size, sequence_length, input_size)
        '''
        (batch_size, sequence_length, hidden_size) = output_grad.shape
        assert_same_shape(output_grad, self.output_)
        hidden_grad = np.zeros_like(self.hn)
        input_grad = np.zeros_like(self.input_)
        for time_stamp in reversed(range(sequence_length)):
            input_grad[:, time_stamp, :], hidden_grad = self.cells[time_stamp].backward(
                output_grad.take(time_stamp, axis=1), hidden_grad, self.params
            )
        return input_grad

    def _clear_gradients(self):
        for key in self.params.keys():
            self.params[key]['deriv'] = np.zeros_like(self.params[key]['deriv'])


class RecurrentNetwork(object):
    '''
    The Model class that takes in inputs and targets and actually trains the network and calculates the loss.
    '''

    def __init__(self, loss: str):
        if loss not in LOSS_FUNCTIONS.keys():
            raise ValueError(f'Loss {loss} is not supported currently')
        self.loss = LOSS_FUNCTIONS[loss]()
        self.layers = []

    def add(self, layer):
        if len(self.layers) == 0:
            self.vocab_size = layer.num_embeddings
        self.layers.append(layer)

    def encode(self, a):
        seq_length = a.shape[0]
        b = np.zeros((seq_length, self.vocab_size))
        for i in range(seq_length):
            b[i, a[i]] = 1
        return b

    def encode_all(self, a):
        return np.stack([self.encode(a_i) for a_i in a])

    def forward(self, x_batch: ndarray):
        '''
        x_batch : (batch_size, sequence_length)
        '''
        x_batch = self.encode_all(x_batch)
        # x_batch : (batch_size, sequence_length, num_embeddings)

        for idx in range(len(self.layers)):
            if idx == 1:
                h0 = np.random.randn(x_batch.shape[0], self.layers[idx].hidden_size)
                x_batch, hn = self.layers[idx].forward(x_batch, h0)
            else:
                x_batch = self.layers[idx].forward(x_batch)
        return x_batch

    def backward(self, loss_grad: ndarray):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
        return loss_grad

    def train(self, x_batch, y_batch):
        lb = LabelBinarizer()
        y_batch = lb.fit_transform(y_batch)

        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        loss_grad = self.loss.backward()
        # for layer in self.layers:
        # layer._clear_gradients()
        self.backward(loss_grad)
        return loss

    def params(self):
        '''
        Returns iterator to layer parameters
        Used for updating parameters in optimization step
        '''
        for idx in range(len(self.layers)):
            if idx == 1:
                params_ = []
                for key in self.layers[idx].params:
                    params_.append(self.layers[idx].params[key]['value'])
                yield from params_
            else:
                if not hasattr(self.layers[idx], 'params'):
                    continue
                yield from self.layers[idx].params

    def param_grads(self):
        '''
        Returns iterator to layer parameter's gradients : dL / d_params
        Used for updating parameters in optimization step
        '''
        for idx in range(len(self.layers)):
            if idx == 1:
                params_grads_ = []
                for key in self.layers[idx].params:
                    params_grads_.append(self.layers[idx].params[key]['deriv'])
                yield from params_grads_
            else:
                yield from self.layers[idx].param_grads

    def predict(self, X: ndarray) -> ndarray:
        pred = self.forward(X)
        return np.argsort(pred, axis=1)[:, -1]


# -------------------- GRU --------------------
class GRUCell(object):
    def __init__(self):
        pass

    def forward(self, x_t, h_t_prev, params):
        '''
        c_t = tanh(W_c[r_t * h_{t - 1}, x_t] + b_c)
        u_t = sigmoid(W_u[h_{t - 1}, x_t] + b_u)
        r_t = sigmoid(W_r[h_{t - 1}, x_t] + b_r)

        h_t = u_t * c_t + (1 - u_t) * h_{t - 1}
        y_t = h_t * W_o + b_o

        Parameters
        ----------
        x_t : ndarray of size = (batch_size, input_size)

        h_t_prev(h_{t - 1}) : ndarray of size = (batch_size, hidden_size)

        Returns
        ----------
        y_t : ndarray of size = (batch_size, hidden_size)
        h_t : ndarray of size = (batch_size, hidden_size)
        '''
        assert_dim(x_t, 2)
        assert_dim(h_t_prev, 2)
        self.x_t = x_t
        self.h_t_prev = h_t_prev
        self.stacked_input = np.column_stack((x_t, h_t_prev))

        # Update gate
        self.update_gate_raw = np.dot(self.stacked_input, params['W_u']) + params['b_u']
        self.update_gate = sigmoid(self.update_gate_raw)

        # Reset gate
        self.reset_gate_raw = np.dot(self.stacked_input, params['W_r']) + params['b_r']
        self.reset_gate = sigmoid(self.reset_gate_raw)

        # Candidate activation unit
        self.stacked_input_reset = np.column_stack(
            (x_t, np.multiply(h_t_prev, self.reset_gate))
        )
        self.cand_act_raw = (
            np.dot(self.stacked_input_reset, params['W_c']) + params['b_c']
        )
        self.cand_act = tanh(self.cand_act_raw)

        # New hidden state
        self.h_t = np.multiply(self.update_gate, self.cand_act) + np.multiply(
            (1 - self.update_gate), self.h_t_prev
        )
        # Output of the cell
        self.y_t = np.dot(h_t, params['W_o']) + params['b_o']
        return self.y_t, self.h_t

    def backward(self, output_grad, hidden_grad, params):
        '''
        c_t = tanh(W_c[r_t * h_{t - 1}, x_t] + b_c)
        u_t = sigmoid(W_u[h_{t - 1}, x_t] + b_u)
        r_t = sigmoid(W_r[h_{t - 1}, x_t] + b_r)

        h_t = u_t * c_t + (1 - u_t) * h_{t - 1}
        y_t = h_t * W_o + b_o

        Parameters
        ----------
        output_grad : ndarray of size = (batch_size, hidden_size)
            dL/dy_t

        hidden_grad : ndarray of size = (batch_size, hidden_size)
            dL/dh_t

        Returns
        ----------
        input_grad : ndarray of size = (batch_size, input_size)
        hidden_input_grad : ndarray of size = (batch_size, hidden_size)
        '''
        assert_same_shape(output_grad, self.y_t)
        assert_same_shape(hidden_grad, self.h_t)

        params['b_o']['deriv'] += output_grad.sum(axis=0)
        params['W_o']['deriv'] += np.dot(self.h_t.T, output_grad)

        d_h_t = np.dot(output_grad, params['W_o']['value'].T)
        d_h_t += hidden_grad

        d_u_t = hidden_grad * (self.cand_act - self.h_t_prev)
        d_u_t_raw = dsigmoid(self.update_gate_raw) * d_u_t

        d_cand_act = hidden_grad * self.update_gate
        d_cand_act_raw = dsigmoid(self.cand_act_raw) * d_cand_act

        d_h_t_prev = (1 - self.update_gate) * hidden_grad

        params['b_u']['deriv'] += np.sum(d_u_t_raw, axis=0)
        params['W_u']['deriv'] += np.dot(self.stacked_input.T, d_u_t_raw)

        params['b_c']['deriv'] += np.sum(d_cand_act_raw, axis=0)
        params['W_c']['deriv'] += np.dot(self.stacked_input_reset.T, d_cand_act_raw)


class GRU(object):
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.first = True
        self._init_parameters()

    def _init_parameters(self):
        '''
        Initialize learning parameters
        '''
        self.params = dict()

        # Candidate Activation
        self.params['W_c'] = dict()
        self.params['b_c'] = dict()
        # Candidate Activation
        self.params['W_c'] = np.random.normal(
            size=(self.input_size + self.hidden_size, self.hidden_size)
        )
        self.params['b_c'] = np.random.normal(size=(1, self.hidden_size))

        # Reset Gate
        self.params['W_r'] = dict()
        self.params['b_r'] = dict()
        # Reset Gate
        self.params['W_r'] = np.random.normal(
            size=(self.input_size + self.hidden_size, self.hidden_size)
        )
        self.params['b_r'] = np.random.normal(size=(1, self.hidden_size))

        # Update Gate
        self.params['W_u'] = dict()
        self.params['b_u'] = dict()
        # Update Gate
        self.params['W_u'] = np.random.normal(
            size=(self.input_size + self.hidden_size, self.hidden_size)
        )
        self.params['b_u'] = np.random.normal(size=(1, self.hidden_size))

        # Output Parameters
        self.params['W_o'] = dict()
        self.params['b_o'] = dict()
        # Output Parameters
        self.params['W_o'] = np.random.normal(size=(self.hidden_size, self.hidden_size))
        self.params['b_o'] = np.random.normal(size=(1, self.hidden_size))

        for key in self.params:
            self.params[key]['deriv'] = np.zeros_like(self.params[key]['value'])

    def forward(self, input_: ndarray, h0: ndarray):
        '''
        Forwards the input and initial hidden state
        Parameters
        ----------
        input_ : ndarray of size = (batch_size, sequence_length, input_size)
            containing the features of the input sequence.

        h0: ndarray of size = (batch_size, hidden_size)
            containing the initial hidden state for each element in the batch

        Returns
        ----------
        output_ : ndarray of size = (batch_size, sequence_length, hidden_size)
            containing the output features (h_t) from the last layer of the RNN, for each t

        hn : ndarray of size = (batch_size, hidden_size)
            final hidden state for each element in the batch
        '''
        pass
