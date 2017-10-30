import theano.tensor as T
import lasagne
import lasagne.layers as L


class GatedAttentionLayerWithQueryAttention(L.MergeLayer):
    # inputs[0]: B * N * 2D
    # inputs[1]: B * Q * 2D
    # inputs[2]: B * Q          (l_m_q)

    def get_output_for(self, inputs, **kwargs):
        M = T.batched_dot(inputs[0], inputs[1].dimshuffle((0,2,1)))             # B x N x Q
        B, N, Q = M.shape
        alphas = T.nnet.softmax(T.reshape(M, (B*N, Q)))
        alphas_r = T.reshape(alphas, (B,N,Q)) * inputs[2].dimshuffle(0, 'x', 1) # B x N x Q
        alphas_r = alphas_r / alphas_r.sum(axis=2, keepdims=True)               # B x N x Q
        q_rep = T.batched_dot(alphas_r, inputs[1])                              # B x N x 2D
        d_gated = inputs[0] * q_rep
        return d_gated

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

class QuerySliceLayer(L.MergeLayer):
    # inputs[0]: B * Q * 2D (q)
    # inputs[1]: B          (q_var)

    def get_output_for(self, inputs, **kwargs):
        q_slice = inputs[0][T.arange(inputs[0].shape[0]), inputs[1], :]     # B x 2D
        return q_slice

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2])

class GatedAttentionLayer(L.MergeLayer):
    # inputs[0]: B * N * 2D
    # inputs[1]: N * 2D

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1].dimshuffle(0, 'x', 1)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


class AttentionSumLayer(L.MergeLayer):
    # inputs[0]: batch * len * h            (d)
    # inputs[1]: batch * h                  (q_slice)
    # inputs[2]: batch * len * num_cand     (c_var)
    # inputs[3]: batch * len                (m_c_var)

    def get_output_for(self, inputs, **kwargs):
        dq = T.batched_dot(inputs[0], inputs[1])    # B x len
        attention = T.nnet.softmax(dq) * inputs[3]  # B x len
        attention = attention / attention.sum(axis=1, keepdims=True)
        probs = T.batched_dot(attention, inputs[2]) # B x num_cand
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[2][0], input_shapes[2][2])

def stack_rnn(l_emb, l_mask, num_layers, num_units,
              grad_clipping=10, dropout_rate=0.,
              bidir=True,
              only_return_final=False,
              name='',
              rnn_layer=lasagne.layers.LSTMLayer):
    """
        Stack multiple RNN layers.
    """

    def _rnn(backwards=True, name=''):
        network = l_emb
        for layer in range(num_layers):
            if dropout_rate > 0:
                network = lasagne.layers.DropoutLayer(network, p=dropout_rate)
            c_only_return_final = only_return_final and (layer == num_layers - 1)
            network = rnn_layer(network, num_units,
                                grad_clipping=grad_clipping,
                                mask_input=l_mask,
                                only_return_final=c_only_return_final,
                                backwards=backwards,
                                name=name + '_layer' + str(layer + 1))
        return network

    network = _rnn(True, name)
    if bidir:
        network = lasagne.layers.ConcatLayer([network, _rnn(False, name + '_back')], axis=-1)
    return network


class AveragePoolingLayer(lasagne.layers.MergeLayer):
    """
        Average pooling.
        incoming: batch x len x h
    """
    def __init__(self, incoming, mask_input=None, **kwargs):
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)
        super(AveragePoolingLayer, self).__init__(incomings, **kwargs)
        if len(self.input_shapes[0]) != 3:
            raise ValueError('the shape of incoming must be a 3-element tuple')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-2] + input_shapes[0][-1:]

    def get_output_for(self, inputs, **kwargs):
        if len(inputs) == 1:
            # mask_input is None
            return T.mean(inputs[0], axis=1)
        else:
            # inputs[0]: batch x len x h
            # inputs[1] = mask_input: batch x len
            return (T.sum(inputs[0] * inputs[1].dimshuffle(0, 1, 'x'), axis=1) /
                    T.sum(inputs[1], axis=1).dimshuffle(0, 'x'))

class orthogonalize(lasagne.layers.MergeLayer):

    """
        A layer to orthogonalize document using soft elimination.
        incomings[0]: batch x h
        incomings[1]: batch x h
        incomings[2]: batch x 4 x h
    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 6:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(orthogonalize, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        self.W_f = self.add_param(init, (self.num_units, self.num_units), name='W_f')
        self.U_f = self.add_param(init, (self.num_units, self.num_units), name='U_f')
        self.V_f = self.add_param(init, (self.num_units, self.num_units), name='V_f')

        self.W_r = self.add_param(init, (self.num_units, self.num_units), name='W_r')
        self.U_r = self.add_param(init, (self.num_units, self.num_units), name='U_r')
        self.V_r = self.add_param(init, (self.num_units, self.num_units), name='V_r')

        self.W_a = self.add_param(init, (self.num_units, self.num_units), name='W_f')
        self.U_a = self.add_param(init, (self.num_units, self.num_units), name='U_f')
        self.w_a = self.add_param(init, (self.num_units, 1), name='V_f')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        # inputs[0]: batch * h
        # inputs[1]: batch * h
        # inputs[2]: batch * h
        # inputs[3]: batch * h
        # inputs[4]: batch * h
        # inputs[5]: batch * h
        '''
        option1 = lasagne.layers.SliceLayer(inputs[2], indices=0, axis=1)
        option2 = lasagne.layers.SliceLayer(inputs[2], indices=1, axis=1)
        option3 = lasagne.layers.SliceLayer(inputs[2], indices=2, axis=1)
        option4 = lasagne.layers.SliceLayer(inputs[2], indices=3, axis=1)
        '''

        
        comm_cond = T.dot(inputs[0], self.W_f) + T.dot(inputs[1], self.V_f)
        
        cond_gate_1 = T.nnet.sigmoid(comm_cond + T.dot(inputs[2], self.U_f))
        cond_gate_2 = T.nnet.sigmoid(comm_cond + T.dot(inputs[3], self.U_f))
        cond_gate_3 = T.nnet.sigmoid(comm_cond + T.dot(inputs[4], self.U_f))
        cond_gate_4 = T.nnet.sigmoid(comm_cond + T.dot(inputs[5], self.U_f))
        
        comm_scale = T.dot(inputs[0], self.W_r) + T.dot(inputs[1], self.V_r)
        scale_gate_1 = T.nnet.sigmoid(comm_scale + T.dot(inputs[2], self.U_r))
        scale_gate_2 = T.nnet.sigmoid(comm_scale + T.dot(inputs[3], self.U_r))
        scale_gate_3 = T.nnet.sigmoid(comm_scale + T.dot(inputs[4], self.U_r))
        scale_gate_4 = T.nnet.sigmoid(comm_scale + T.dot(inputs[5], self.U_r))

        similarity1 = T.sum(inputs[0]*inputs[2], axis=1, keepdims=True)
        similarity2 = T.sum(inputs[0]*inputs[3], axis=1, keepdims=True)
        similarity3 = T.sum(inputs[0]*inputs[4], axis=1, keepdims=True)
        similarity4 = T.sum(inputs[0]*inputs[5], axis=1, keepdims=True)
        
        norm = inputs[0].norm(L=2, axis=1)**2
	norm = norm.reshape((inputs[0].shape[0],1))
        #temp = similarity1/(norm + 1e-12) 
        component1 = (similarity1/(norm + 1e-12))*inputs[0]
        component2 = (similarity2/(norm + 1e-12))*inputs[0]
        component3 = (similarity3/(norm + 1e-12))*inputs[0]
        component4 = (similarity4/(norm + 1e-12))*inputs[0]

        new_rep_1 = (1 + scale_gate_1*cond_gate_1 - scale_gate_1)*inputs[0] - (scale_gate_1*(2*cond_gate_1 - 1))*component1
        new_rep_2 = (1 + scale_gate_2*cond_gate_2 - scale_gate_2)*inputs[0] - (scale_gate_2*(2*cond_gate_2 - 1))*component2
        new_rep_3 = (1 + scale_gate_3*cond_gate_3 - scale_gate_3)*inputs[0] - (scale_gate_3*(2*cond_gate_3 - 1))*component3
        new_rep_4 = (1 + scale_gate_4*cond_gate_4 - scale_gate_4)*inputs[0] - (scale_gate_4*(2*cond_gate_4 - 1))*component4

         
        att_1 = T.tanh(T.dot(new_rep_1, self.W_a) + T.dot(inputs[2], self.U_a))
        att_2 = T.tanh(T.dot(new_rep_2, self.W_a) + T.dot(inputs[3], self.U_a))
        att_3 = T.tanh(T.dot(new_rep_3, self.W_a) + T.dot(inputs[4], self.U_a))
        att_4 = T.tanh(T.dot(new_rep_4, self.W_a) + T.dot(inputs[5], self.U_a))

        w_att_1 = T.sum(T.dot(att_1, T.reshape(self.w_a, (self.num_units, 1))),axis=1, keepdims=True)
        w_att_2 = T.sum(T.dot(att_2, T.reshape(self.w_a, (self.num_units, 1))),axis=1, keepdims=True)
        w_att_3 = T.sum(T.dot(att_3, T.reshape(self.w_a, (self.num_units, 1))),axis=1, keepdims=True)
        w_att_4 = T.sum(T.dot(att_4, T.reshape(self.w_a, (self.num_units, 1))),axis=1, keepdims=True)

        att_norm = T.nnet.softmax(T.concatenate([w_att_1, w_att_2, w_att_3, w_att_4], axis=1))

        #att_norm_1 = T.reshape(lasagne.layers.SliceLayer(att_norm, indices=slice(0,1), axis=1),)
        #att_norm_2 = lasagne.layers.SliceLayer(att_norm, indices=slice(1,2), axis=1)
        #att_norm_3 = lasagne.layers.SliceLayer(att_norm, indices=slice(2,3), axis=1)
        #att_norm_4 = lasagne.layers.SliceLayer(att_norm, indices=slice(3,4), axis=1)
        att_norm_t = att_norm.dimshuffle(1,0)
        att_norm_1 = att_norm_t[0:1].dimshuffle(1,0)
        att_norm_2 = att_norm_t[1:2].dimshuffle(1,0)
        att_norm_3 = att_norm_t[2:3].dimshuffle(1,0)
        att_norm_4 = att_norm_t[3:4].dimshuffle(1,0)
        new_rep = att_norm_1.repeat(self.num_units, axis=1)*new_rep_1 + att_norm_2.repeat(self.num_units, axis=1)*new_rep_2 + att_norm_3.repeat(self.num_units, axis=1)*new_rep_3 + att_norm_4.repeat(self.num_units, axis=1)*new_rep_4
             
        return new_rep


class MLPAttentionLayer(lasagne.layers.MergeLayer):
    """
        An MLP attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
        Reference: http://arxiv.org/abs/1506.03340
    """
    def __init__(self, incomings, num_units,
                 nonlinearity=lasagne.nonlinearities.tanh,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(MLPAttentionLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_units = num_units
        self.W0 = self.add_param(init, (self.num_units, self.num_units), name='W0_mlp')
        self.W1 = self.add_param(init, (self.num_units, self.num_units), name='W1_mlp')
        self.Wb = self.add_param(init, (self.num_units, ), name='Wb_mlp')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        M = T.dot(inputs[0], self.W0) + T.dot(inputs[1], self.W1).dimshuffle(0, 'x', 1)
        M = self.nonlinearity(M)
        alpha = T.nnet.softmax(T.dot(M, self.Wb))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / alpha.sum(axis=1).reshape((alpha.shape[0], 1))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)


class LengthLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return T.cast(input.sum(axis=-1) - 1, 'int32')

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]

class QuerySliceLayer(L.MergeLayer):
    # inputs[0]: B * Q * 2D (q)
    # inputs[1]: B          (q_var)

    def get_output_for(self, inputs, **kwargs):
        q_slice = inputs[0][T.arange(inputs[0].shape[0]), inputs[1], :]     # B x 2D
        return q_slice

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2])

class BilinearAttentionLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearAttentionLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):

        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # W: h * h

        M = T.dot(inputs[1], self.W).dimshuffle(0, 'x', 1)
        alpha = T.nnet.softmax(T.sum(inputs[0] * M, axis=2))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / alpha.sum(axis=1).reshape((alpha.shape[0], 1))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)

class BilinearDotLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearDotLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):

        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # W: h * h
        M = T.dot(inputs[1], self.W).dimshuffle(0, 'x', 1) #batch * 1 * h
        alpha = T.nnet.softmax(T.sum(inputs[0] * M, axis=2)) #batch * len
        return alpha

class BilinearDotLayerTensor(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x len x h

    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearDotLayerTensor, self).__init__(incomings, **kwargs)
        self.num_units = num_units

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):
        alpha = T.nnet.softmax(T.sum(inputs[0] * inputs[1], axis=2))
        return alpha

class DotProductAttentionLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
    """
    def __init__(self, incomings, mask_input=None, **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(DotProductAttentionLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):

        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # mask_input (if any): batch * len

        alpha = T.nnet.softmax(T.sum(inputs[0] * inputs[1].dimshuffle(0, 'x', 1), axis=2))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / alpha.sum(axis=1).reshape((alpha.shape[0], 1))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)
