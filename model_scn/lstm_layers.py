
#import numpy as np
import theano
import theano.tensor as tensor
from utils import _p, numpy_floatX
from utils import uniform_weight, zero_bias

""" Encoder using LSTM Recurrent Neural Network. """

def param_init_encoder(options, params, prefix='encoder_lstm'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    n_f = options['n_f']
    n_y = options['n_y']
    
    params[_p(prefix, 'Wa_i')] = uniform_weight(n_x,n_f)
    params[_p(prefix, 'Wa_f')] = uniform_weight(n_x,n_f)
    params[_p(prefix, 'Wa_o')] = uniform_weight(n_x,n_f)
    params[_p(prefix, 'Wa_c')] = uniform_weight(n_x,n_f)
    
    params[_p(prefix, 'Wb_i')] = uniform_weight(n_y,n_f)
    params[_p(prefix, 'Wb_f')] = uniform_weight(n_y,n_f)
    params[_p(prefix, 'Wb_o')] = uniform_weight(n_y,n_f)
    params[_p(prefix, 'Wb_c')] = uniform_weight(n_y,n_f)
    
    params[_p(prefix, 'Wc_i')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Wc_f')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Wc_o')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Wc_c')] = uniform_weight(n_h,n_f)
    
    params[_p(prefix, 'Ua_i')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Ua_f')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Ua_o')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Ua_c')] = uniform_weight(n_h,n_f)
    
    params[_p(prefix, 'Ub_i')] = uniform_weight(n_y,n_f)
    params[_p(prefix, 'Ub_f')] = uniform_weight(n_y,n_f)
    params[_p(prefix, 'Ub_o')] = uniform_weight(n_y,n_f)
    params[_p(prefix, 'Ub_c')] = uniform_weight(n_y,n_f)
    
    params[_p(prefix, 'Uc_i')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Uc_f')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Uc_o')] = uniform_weight(n_h,n_f)
    params[_p(prefix, 'Uc_c')] = uniform_weight(n_h,n_f)
    
    params[_p(prefix,'b_i')] = zero_bias(n_h)
    params[_p(prefix,'b_f')] = zero_bias(n_h)
    params[_p(prefix,'b_o')] = zero_bias(n_h)
    params[_p(prefix,'b_c')] = zero_bias(n_h)
    
    return params
    
def encoder_layer(tparams, state_below, mask, y, seq_output=True, prefix='encoder_lstm'):
    
    """ state_below: size of  n_steps * n_samples * n_x
    """

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]

    n_h = tparams[_p(prefix,'Ua_i')].shape[0]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]
        
    tmp1_i = tensor.dot(state_below, tparams[_p(prefix, 'Wa_i')]) 
    tmp1_f = tensor.dot(state_below, tparams[_p(prefix, 'Wa_f')])
    tmp1_o = tensor.dot(state_below, tparams[_p(prefix, 'Wa_o')])
    tmp1_c = tensor.dot(state_below, tparams[_p(prefix, 'Wa_c')])
    
    tmp2_i = tensor.dot(y, tparams[_p(prefix, 'Wb_i')]).dimshuffle('x',0,1)
    tmp2_f = tensor.dot(y, tparams[_p(prefix, 'Wb_f')]).dimshuffle('x',0,1)
    tmp2_o = tensor.dot(y, tparams[_p(prefix, 'Wb_o')]).dimshuffle('x',0,1)
    tmp2_c = tensor.dot(y, tparams[_p(prefix, 'Wb_c')]).dimshuffle('x',0,1)
    
    # n_steps * n_sample * n_h
    state_below_i = tensor.dot(tmp1_i*tmp2_i,tparams[_p(prefix, 'Wc_i')].T) + tparams[_p(prefix, 'b_i')] 
    state_below_f = tensor.dot(tmp1_f*tmp2_f,tparams[_p(prefix, 'Wc_f')].T) + tparams[_p(prefix, 'b_f')] 
    state_below_o = tensor.dot(tmp1_o*tmp2_o,tparams[_p(prefix, 'Wc_o')].T) + tparams[_p(prefix, 'b_o')]  
    state_below_c = tensor.dot(tmp1_c*tmp2_c,tparams[_p(prefix, 'Wc_c')].T) + tparams[_p(prefix, 'b_c')]                   

    def _step(m_, x_i, x_f, x_o, x_c, h_, c_, Ua_i, Ua_f, Ua_o, Ua_c, Ub_i, Ub_f, Ub_o, Ub_c, Uc_i, Uc_f, Uc_o, Uc_c,y):
        preact_i = tensor.dot(h_, Ua_i) * (tensor.dot(y, Ub_i))
        preact_i = tensor.dot(preact_i,Uc_i.T) + x_i
        
        preact_f = tensor.dot(h_, Ua_f) * (tensor.dot(y, Ub_f))
        preact_f = tensor.dot(preact_f,Uc_f.T) + x_f
        
        preact_o = tensor.dot(h_, Ua_o) * (tensor.dot(y, Ub_o))
        preact_o = tensor.dot(preact_o,Uc_o.T) + x_o
        
        preact_c = tensor.dot(h_, Ua_c) * (tensor.dot(y, Ub_c))
        preact_c = tensor.dot(preact_c,Uc_c.T) + x_c
        
        i = tensor.nnet.sigmoid(preact_i)
        f = tensor.nnet.sigmoid(preact_f)
        o = tensor.nnet.sigmoid(preact_o)
        c = tensor.tanh(preact_c)
        
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    seqs = [mask, state_below_i,state_below_f,state_below_o,state_below_c]
    non_seqs = [tparams[_p(prefix, 'Ua_i')],tparams[_p(prefix, 'Ua_f')],tparams[_p(prefix, 'Ua_o')],tparams[_p(prefix, 'Ua_c')],
                tparams[_p(prefix, 'Ub_i')],tparams[_p(prefix, 'Ub_f')],tparams[_p(prefix, 'Ub_o')],tparams[_p(prefix, 'Ub_c')],
                tparams[_p(prefix, 'Uc_i')],tparams[_p(prefix, 'Uc_f')],tparams[_p(prefix, 'Uc_o')],tparams[_p(prefix, 'Uc_c')],
                y]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                    n_samples,n_h),
                                              tensor.alloc(numpy_floatX(0.),
                                                    n_samples,n_h)],
                                non_sequences = non_seqs,
                                name=_p(prefix, '_layers'),
                                n_steps=n_steps,
                                strict=True)
    
    h_rval = rval[0] 
    if seq_output:
        return h_rval
    else:
        # size of n_samples * n_h
        return h_rval[-1]    
