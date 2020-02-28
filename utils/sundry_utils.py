
class LinearAnnealing:
    def __init__(self, x, x_, end):
        '''
        Params: 
            x: start value
            x_: end value
            end: annealing time
        '''
        assert end != 0, 'the time steps for annealing must larger than 0.'
        self.x = x
        self.x_ = x_
        self.interval = (x_ - x) / end

    def __call__(self, current):
        return max(self.x + self.interval * current, self.x_)