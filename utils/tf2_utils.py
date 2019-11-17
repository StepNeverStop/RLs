import tensorflow as tf

def show_graph(name='my_func_trace'):
    '''
    show tf2 graph in tensorboard. work for ppo, have bug in off-policy algorithm, like dqn..
    TODO: fix bug when showing graph of off-policy algorithm based on TF2.
    '''
    def show_tf2_graph(func):
        def inner(*args, **kwargs):
            tf.summary.trace_on(graph=True)
            ret = func(*args, **kwargs)
            tf.summary.trace_export(name=name)
            return ret
        return inner
    return show_tf2_graph