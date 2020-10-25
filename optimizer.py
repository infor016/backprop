import numpy as np


class Optimizer:
    """
    Attribut

    :param trainable_layers: list
        trainable layer dengan nilai bobot dan bias
    """
    def __init__(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def init(self):
        """
        inisialisasi optimizer
        """
        raise NotImplementedError

    def update(self, learning_rate, w_grad, b_grad, step):
        """
        Update parameter dari trainable layer

        parameter
        :param learning_rate: float
            ukuran nilai dari lr digunakan untuk update tiap step
        :param w_grad: np.array
            gradient bobot
        :param b_grad: np.array
            gradient bias
        :param step: int
            berapa banyak update yang dilakukan
        """
        raise NotImplementedError
        

class GradientDescent(Optimizer):
    def __init__(self, trainable_layers):
        Optimizer.__init__(self, trainable_layers)

    def init(self):
        pass

    def update(self, learning_rate, w_grad, b_grad, step):
        for layer in self.trainable_layers:
            layer.update_parameters(dw=learning_rate * w_grad[layer],
                                    db=learning_rate * b_grad[layer])


class RMSProp(Optimizer):
    def __init__(self, trainable_layers, beta=0.9, epsilon=1e-8):
        Optimizer.__init__(self, trainable_layers)
        self.s = {}
        self.beta = beta
        self.epsilon = epsilon

    def init(self):
        for layer in self.trainable_layers:
            w, b = layer.get_parameters()
            w_shape = w.shape
            b_shape = b.shape
            self.s[('dw', layer)] = np.zeros(w_shape)
            self.s[('db', layer)] = np.zeros(b_shape)

    def update(self, learning_rate, w_grad, b_grad, step):
        s_corrected = {}
        s_correction_term = 1 - np.power(self.beta, step)

        for layer in self.trainable_layers:
            layer_dw = ('dw', layer)
            layer_db = ('db', layer)

            self.s[layer_dw] = (self.beta * self.s[layer_dw] + (1 - self.beta) * np.square(w_grad[layer]))
            self.s[layer_db] = (self.beta * self.s[layer_db] + (1 - self.beta) * np.square(b_grad[layer]))

            s_corrected[layer_dw] = self.s[layer_dw] / s_correction_term
            s_corrected[layer_db] = self.s[layer_db] / s_correction_term

            dw = (learning_rate * (w_grad[layer] / (np.sqrt(s_corrected[layer_dw]) + self.epsilon)))
            db = (learning_rate * (b_grad[layer] / (np.sqrt(s_corrected[layer_db]) + self.epsilon)))

            layer.update_parameters(dw, db)


class Adam(Optimizer):
    def __init__(self, trainable_layers, beta1=0.9, beta2=0.999, epsilon=1e-8):
        Optimizer.__init__(self, trainable_layers)
        self.v = {}
        self.s = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def init(self):
        for layer in self.trainable_layers:
            w, b = layer.get_parameters()
            w_shape = w.shape
            b_shape = b.shape
            self.v[('dw', layer)] = np.zeros(w_shape)
            self.v[('db', layer)] = np.zeros(b_shape)
            self.s[('dw', layer)] = np.zeros(w_shape)
            self.s[('db', layer)] = np.zeros(b_shape)

    def update(self, learning_rate, w_grad, b_grad, step):
        v_corrected = {}
        s_corrected = {}
        v_correction_term = 1 - np.power(self.beta1, step)
        s_correction_term = 1 - np.power(self.beta2, step)

        for layer in self.trainable_layers:
            layer_dw = ('dw', layer)
            layer_db = ('db', layer)

            self.v[layer_dw] = (self.beta1 * self.v[layer_dw] + (1 - self.beta1) * w_grad[layer])
            self.v[layer_db] = (self.beta1 * self.v[layer_db] + (1 - self.beta1) * b_grad[layer])

            v_corrected[layer_dw] = self.v[layer_dw] / v_correction_term
            v_corrected[layer_db] = self.v[layer_db] / v_correction_term

            self.s[layer_dw] = (self.beta2 * self.s[layer_dw] + (1 - self.beta2) * np.square(w_grad[layer]))
            self.s[layer_db] = (self.beta2 * self.s[layer_db] + (1 - self.beta2) * np.square(b_grad[layer]))

            s_corrected[layer_dw] = self.s[layer_dw] / s_correction_term
            s_corrected[layer_db] = self.s[layer_db] / s_correction_term

            dw = (learning_rate * v_corrected[layer_dw] / (np.sqrt(s_corrected[layer_dw]) + self.epsilon))
            db = (learning_rate * v_corrected[layer_db] / (np.sqrt(s_corrected[layer_db]) + self.epsilon))

            layer.update_parameters(dw, db)
