import numpy as np
import sys


class BoundingBox(object):


    def __init__(self, lower=None, upper=None, k=None, all_space=False):
 
        if lower is not None:
            self.lower = np.array(lower)
            self.upper = np.array(upper) if upper is not None else self.lower
        elif k is not None:
            if all_space:
                self.lower = np.full(k, sys.float_info.min)
                self.upper = np.full(k, sys.float_info.max)
            else:
                self.lower = np.full(k, sys.float_info.max)
                self.upper = np.full(k, sys.float_info.min)
        else:
            self.lower = None
            self.upper = None

    def intersection(self, other):

        lower = np.maximum(self.lower, other.lower)
        upper = np.minimum(self.upper, other.upper)
        return BoundingBox(lower=lower, upper=upper)

    def union(self, other):
   
        lower = np.minimum(self.lower, other.lower)
        upper = np.maximum(self.upper, other.upper)
        return BoundingBox(lower=lower, upper=upper)

    def split(self, dim, value):
 
        left = BoundingBox(lower=np.copy(self.lower),
                           upper=np.copy(self.upper))
        left.upper[dim] = value
        right = BoundingBox(lower=np.copy(self.lower),
                            upper=np.copy(self.upper))
        right.lower[dim] = value
        return left, right

    def expand(self, eps=0, how='add'):
  
        if how == 'add':
            return BoundingBox(self.lower - eps, self.upper + eps)
        elif how == 'multiply':
            span = self.upper - self.lower
            return BoundingBox(self.lower - eps * span,
                               self.upper + eps * span)

    def contains(self, vector):
 
        return np.all(self.lower <= vector) and np.all(self.upper >= vector)

    def __repr__(self):
        return 'BoundingBox(lower=%s\n\tupper=%s)' % (
            str(self.lower), str(self.upper))
