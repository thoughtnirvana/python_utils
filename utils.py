#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
"""
Provide some widely useful utilities. Safe for "from utils import *".
Originally taken from Peter Norwig's utils.py
http://code.google.com/p/aima-python/source/browse/trunk/utils.py
"""
from __future__ import print_function
import operator, math, random, sys, heapq, re
import logging
import subprocess as sp
import unicodedata, string
from operator import itemgetter
from heapq import nlargest
from itertools import *
from functools import wraps
from collections import deque, defaultdict

class Struct:
    """Create an instance with argument=value slots.
    This is for making a lightweight object whose class doesn't matter."""
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __cmp__(self, other):
        if isinstance(other, Struct):
            return cmp(self.__dict__, other.__dict__)
        else:
            return cmp(self.__dict__, other)

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k, v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(args)

def update(x, **entries):
    """Update a dict; or an object with slots; according to entries.
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    >>> update(Struct(a=1), a=10, b=20)
    Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries)
    return x

# Functions on Sequences
# NOTE: Sequence functions (count_if, find_if, every, some) take function
# argument first (like reduce, filter, and map).

def removeall(item, seq):
    """Return a copy of seq (or string) with all occurences of item removed.
    >>> removeall(3, [1, 2, 3, 3, 2, 1, 3])
    [1, 2, 2, 1]
    >>> removeall(4, [1, 2, 3])
    [1, 2, 3]
    """
    if isinstance(seq, str):
      return seq.replace(item, '')
    else:
      return [x for x in seq if x != item]

def remove_index(index, seq):
    """
    Returns seq without the given indexes.
    >>> remove_index(-1, [1,2,3])
    [1, 2]
    >>> remove_index([0,-1], [1,2,3,4,5,6])
    [2, 3, 4, 5]
    """
    length = len(seq)
    if not isinstance(index, list): index = [index]
    exclude_indexes = [list_index(i, len(seq)) for i in index]
    return [v for i,v in enumerate(seq)
                if i not in exclude_indexes]

def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements.
    >>> unique([1, 2, 3, 2, 1])
    [1, 2, 3]
    """
    return list(set(seq))

def product(numbers):
    """Return the product of the numbers.
    >>> product([1,2,3,4])
    24
    """
    return reduce(operator.mul, numbers, 1)

def count_if(predicate, seq):
    """Count the number of elements of seq for which the predicate is true.
    >>> count_if(callable, [42, None, max, min])
    2
    """
    f = lambda count, x: count + (not not predicate(x))
    return reduce(f, seq, 0)

def find_if(predicate, seq):
    """If there is an element of seq that satisfies predicate; return it.
    >>> find_if(callable, [3, min, max])
    <built-in function min>
    >>> find_if(callable, [1, 2, 3])
    """
    for x in seq:
        if predicate(x): return x
    return None

def every(predicate, seq):
    """True if every element of seq satisfies predicate.
    >>> every(callable, [min, max])
    1
    >>> every(callable, [min, 3])
    0
    """
    for x in seq:
        if not predicate(x): return False
    return True

def some(predicate, seq):
    """If some element x of seq satisfies predicate(x), return predicate(x).
    >>> some(callable, [min, 3])
    1
    >>> some(callable, [2, 3])
    0
    """
    for x in seq:
        px = predicate(x)
        if  px: return px
    return False

def isin(elt, seq):
    """Like (elt in seq), but compares with is, not ==.
    >>> e = []; isin(e, [1, e, 3])
    True
    >>> isin(e, [1, [], 3])
    False
    """
    for x in seq:
        if elt is x: return True
    return False

# Functions on sequences of numbers
# NOTE: these take the sequence argument first, like min and max,
# and like standard math notation: \sigma (i = 1..n) fn(i)
# A lot of programing is finding the best value that satisfies some condition;
# so there are three versions of argmin/argmax, depending on what you want to
# do with ties: return the first one, return them all, or pick at random.

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def argmin_list(seq, fn):
    """Return a list of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """
    best_score, best = fn(seq[0]), []
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = [x], x_score
        elif x_score == best_score:
            best.append(x)
    return best

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                    best = x
    return best

def argmax_list(seq, fn):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    return argmin_list(seq, lambda x: -fn(x))

def argmax_random_tie(seq, fn):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmin_random_tie(seq, lambda x: -fn(x))


# Statistical and mathematical functions

def histogram(values, mode=0, bin_function=None):
    """Return a list of (value, count) pairs, summarizing the input values.
    Sorted by increasing value, or if mode=1, by decreasing count.
    If bin_function is given, map it over values first."""
    if bin_function: values = map(bin_function, values)
    bins = {}
    for val in values:
        bins[val] = bins.get(val, 0) + 1
    if mode:
        return sorted(bins.items(), key=lambda v: v[1], reverse=True)
    else:
        return sorted(bins.items())

def dtanh(y):
    return 1.0 - y*y

def log2(x):
    """Base 2 logarithm.
    >>> log2(1024)
    10.0
    """
    return math.log10(x) / math.log10(2)

def mode(values):
    """Return the most common value in the list of values.
    >>> mode([1, 2, 3, 2])
    2
    """
    return histogram(values, mode=1)[0][0]

def median(values):
    """Return the middle value, when the values are sorted.
    If there are an odd number of elements, try to average the middle two.
    If they can't be averaged (e.g. they are strings), choose one at random.
    >>> median([10, 100, 11])
    11
    >>> median([1, 2, 3, 4])
    2.5
    """
    n = len(values)
    values = sorted(values)
    if n % 2 == 1:
        return values[n/2]
    else:
        middle2 = values[(n/2)-1:(n/2)+1]
        try:
            return mean(middle2)
        except TypeError:
            return random.choice(middle2)

def mean(values):
    """Return the arithmetic average of the values."""
    return sum(values) / float(len(values))

def weighted_mean(weighted_values):
    """
    Returns the weighted mean of the given (weight, value) list.
    """
    vals = 0; weights =0
    for weight, value in weighted_values:
        vals += weight * value
        weights += weight
    return vals/float(weights)

def stddev(values, meanval=None):
    """The standard deviation of a set of values.
    Pass in the mean if you already know it."""
    if meanval == None: meanval = mean(values)
    return math.sqrt(variance(values, meanval))

def variance(values, meanval=None):
    """The variance of a set of values.
    Pass in the mean if you already know it."""
    if meanval == None: meanval = mean(values)
    return sum([(x - meanval)**2 for x in values]) / len(values)

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum(x * y for x, y in zip(X, Y))

def vector_add(a, b):
    """Component-wise addition of two vectors.
    >>> vector_add((0, 1), (8, 9))
    (8, 10)
    """
    return tuple(map(operator.add, a, b))

def probability(p):
    "Return true with probability p."
    return p > random.uniform(0.0, 1.0)

def num_or_str(x):
    """The argument is a string; convert to a number if possible, or strip it.
    >>> num_or_str('42')
    42
    >>> num_or_str(' 42x ')
    '42x'
    """
    if isnumber(x): return x
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
                return str(x).strip()

def normalize(numbers, total=1.0):
    """Multiply each number by a constant such that the sum is 1.0 (or total).
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    k = total / sum(numbers)
    return [k * n for n in numbers]

def distance((ax, ay), (bx, by)):
    "The distance between two (x, y) points."
    return math.hypot((ax - bx), (ay - by))

def distance2((ax, ay), (bx, by)):
    "The square of the distance between two (x, y) points."
    return (ax - bx)**2 + (ay - by)**2

def euclid_distance(predictions, targets):
    """
    Generalized Euclid's distance to calculate the distance between 2 vectors.
    """
    return math.sqrt(sum((p - t)**2 for p, t in zip(predictions, targets)))

def rms_error(predictions, targets):
    return math.sqrt(ms_error(predictions, targets))

def ms_error(predictions, targets):
    return mean([(p - t)**2 for p, t in zip(predictions, targets)])

def mean_error(predictions, targets):
    return mean([abs(p - t) for p, t in zip(predictions, targets)])

def mean_boolean_error(predictions, targets):
    return mean([(p != t)   for p, t in zip(predictions, targets)])

def clip(vector, lowest, highest):
    """Return vector, except if any element is less than the corresponding
    value of lowest or more than the corresponding value of highest, clip to
    those values.
    >>> clip((-1, 10), (0, 0), (9, 9))
    (0, 9)
    """
    return type(vector)(map(min, map(max, vector, lowest), highest))

def uniquecounts(rows, target=-1):
    """
    Create counts of possible results.
    """
    results = defaultdict(lambda: int(0))
    # Record the count for all results.
    for row in rows: results[row[target]] += 1
    return results

def entropy(rows, target=-1):
    """
    Entropy is the sum of p(x)log(p(x)) across all
    the different possible results
    It should be zero if all rows have the same result.
    The default result is the last column.
    """
    results = uniquecounts(rows, target)
    # Now calculate the entropy
    ent = 0.0
    total_len = len(rows)
    for r in results.keys():
        p = float(results[r]) / total_len
        ent -= p* log2(p)
    return ent

# Misc Functions

def get_err_logger(label):
    """
    Returns a logger object which logs
    on stderr.
    """
    # Get logger.
    logger = logging.getLogger(name(label))
    logger.setLevel(logging.DEBUG)
    # Get handler.
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    # Create formatter and setup logger.
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def list_index(idx, length):
    """
    Converts negative list indexes to actual indexes.
    >>> list_index(-1, 3)
    2
    """
    if idx < 0: return length + idx
    else: return idx

def web_tokenize(text):
    """
    Tokenizes the given text. Returns the list of words.
    Smiles are treated as separate tokens. Punctuations are
    removed from words - 'happy' and 'happy.' yield the same
    token 'happy'
    >>> web_tokenize("Had an amazing day:). Super happy.")
    [':)', 'had', 'an', 'amazing', 'day', 'super', 'happy']
    >>> web_tokenize("I am so fucked up:(")
    [':(', 'am', 'so', 'fucked', 'up']
    >>> web_tokenize("That was idiotic, stupid and retarded!")
    ['that', 'was', 'idiotic', 'stupid', 'and', 'retarded']
    >>> web_tokenize("How idiotic of you?:(")
    [':(', 'how', 'idiotic', 'of', 'you']
    """
    # List of smileys. Smileys should be considered separate tokens
    smileys = """
        :-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
        :D C: ():-D :D 8D xD XD =D =3 <=3 <=8
        --!-- :-( :( :c :< :[ :{ D: D8 D; D= DX v.v
        :-O :O O_O o_o 8O OwO O-O 0_o O_o O3O o0o ;o_o; o...o 0w0
        :-9 ;-) ;) *) ;] ;D :-P :P XP :-p :p =p :-b :b
        :-/ :/ :\ =/ =\ :S :| d:-) qB-) :)~ :-)>.... :-X :X :-# :#
        O:-) 0:3 O:  :'( ;*( T_T TT_TT T.T :-* :* ^o)
        >:) >;) >:-) B) B-) 8) 8-) ^>.>^ ^<.<^ ^>_>^ ^<_<^
        D:< >:( D-:< >:-( :-@ ;( `_´ D< <3 <333 </3
        =^_^= =>.>= =<_<= =>.<=	\,,/
        \o/	\o o/ d'-' d'_' d'-'b d'_'b
        ) o/\o :& :u @}-;-'--- d^_^b d-_-b (^_^) (^-^) (^ ^) (^.^) ???
        (~_^) (^_~) ~.^ ^.~	(>_<) (>.<)	(>_>) (¬_¬)	(-_-) (^o^)
        (^3^) (^_^') ^_^_^') ^^" ^^^_.^') ^^_^^; ^&^^.^;& ^^^; ^^^7
        d(>w<)b	q(;^;)p	(;_;) (T_T) (T~T) (ToT) (T^T) (._.) (,_,)
        ［(－－)］ZZzzz... eX_X) x_x O?O &_& 0-0 (^^^)(\_/) B)B(
    """
    tokens = []
    smiley_tokens = []
    smiley_pats = '|'.join(map(re.escape,
                            filter(lambda x: x,
                                re.split(r'\s+', smileys))))
    for word in re.split(r'\s+', text):
        # Tokenize smileys. Remove them from regular words and add them to tokens.
        for smiley in re.findall(smiley_pats, word):
            word = word.replace(smiley, '')
            smiley_tokens.append(smiley)
        # Don't append null strings.
        if word: tokens.append(word)
    # Remove punctuations at the beginning or the end.
    # We don't want to treat 'happy' and 'happy.' differently.
    punct_pat = '|'.join(
                    map(lambda s: r'^{0}|{0}$'.format( re.escape(s) ),
                        string.punctuation.replace('#', '')))
    tokens = (re.sub(punct_pat, '', t) for t in tokens)
    # Remove tokens less than 2 characters or greater than 20.
    # Lowercase all tokens for modelling.
    tokens = (t.lower() for t in tokens
                if 2<=len(t)<20)
    # Return smileys and regular words.
    return list(chain(smiley_tokens, tokens))

def printf(format, *args):
    """Format args with the first argument as format string, and write.
    Return the last arg, or format itself if there are no args."""
    sys.stdout.write(str(format) % args)
    return if_(args, args[-1], format)

def caller(n=1):
    """Return the name of the calling function n levels up in the frame stack.
    >>> caller(0)
    'caller'
    >>> def f():
    ...     return caller()
    >>> f()
    'f'
    """
    import inspect
    return  inspect.getouterframes(inspect.currentframe())[n][3]

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        @wraps(fn)
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @wraps(fn)
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]
        memoized_fn.cache = {}
    return memoized_fn

def singleton(klass):
    """
    Converts a class to singleton. To be used as a class decorator.
    >>> @singleton
    ... class Foo: pass
    ...
    >>> f = Foo()
    >>> f.a = 1
    >>> f2 = Foo()
    >>> print(f2.a)
    1
    """
    @wraps(klass)
    def on_call(*args, **kwargs):
        if on_call.instance is None:
            on_call.instance = klass(*args, **kwargs)
        return on_call.instance
    on_call.instance = None
    return on_call

def borg(fn):
    """
    Wraps the init method to make the class a borg.
    This decorators should be applied to the __init__ method.
    >>> class Bar(object):
    ...     @borg
    ...     def __init__(self, val):
    ...             self.val = val
    ...
    >>> b = Bar(3)
    >>> c = Bar(4)
    >>> b.val
    4
    >>> c.val
    4
    """
    @wraps(fn)
    def on_init(*args, **kwargs):
        args[0].__dict__ = on_init.shared_dict
        return fn(*args, **kwargs)
    on_init.shared_dict = {}
    return on_init

def if_(test, result, alternative):
    """Like C++ and Java's (test ? result : alternative), except
    both result and alternative are always evaluated. However, if
    either evaluates to a function, it is applied to the empty arglist,
    so you can delay execution by putting it in a lambda.
    >>> if_(2 + 2 == 4, 'ok', lambda: expensive_computation())
    'ok'
    """
    if fn_expr(test):
        return fn_expr(result)
    else:
        return fn_expr(alternative)

def fn_wrap(fn, *args, **kwargs):
    """
    Wraps the givne functions and arguments in a lambda.
    Useful for delaying execution like passing params.
    """
    return lambda: fn(*args, **kwargs)

def fn_expr(expr):
    """
    If expr is callable, calls and returns the result.
    Else returns it as is.
    >>> fn_expr(2)
    2
    >>> fn_expr(2+2)
    4
    >>> fn_expr(2 + 2 == 4)
    True
    >>> fn_expr(lambda: 2 + 2)
    4
    """
    if callable(expr): return expr()
    return expr

def name(object):
    "Try to find some reasonable name for the object."
    return (getattr(object, 'name', 0) or getattr(object, '__name__', 0)
            or getattr(getattr(object, '__class__', 0), '__name__', 0)
            or str(object))

def isnumber(x):
    "Is x a number? We say it is if it has a __int__ method."
    return hasattr(x, '__int__') or hasattr(x, '__float__')

def issequence(x):
    "Is x a sequence? We say it is if it has a __getitem__ method."
    return hasattr(x, '__getitem__')

def print_table(table, header=None, sep=' ', numfmt='%g'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '%6.2f'.
    (If you want different formats in differnt columns, don't use print_table.)
    sep is the separator between columns."""
    justs = [if_(isnumber(x), 'rjust', 'ljust') for x in table[0]]
    if header:
        table = [header] + table
    table = [[if_(isnumber(x), lambda: numfmt % x, x)  for x in row]
             for row in table]
    maxlen = lambda seq: max(map(len, seq))
    sizes = map(maxlen, zip(*[map(str, row) for row in table]))
    for row in table:
        for (j, size, x) in zip(justs, sizes, row):
            print(getattr(str(x), j)(size), sep,)
        print("\n")

def exec_cmd(cmd):
    "Executes given command with bash and returns output."
    output = None
    try:
        output = sp.Popen(["/bin/bash", "-c", cmd], stdout=sp.PIPE).communicate()[0]
    except OSError, e:
        print("Execution failed", e, file=sys.stderr)
    finally:
        return output

def sys_cmd(cmd):
    """
    Executes given command and returns the status code.
    """
    try:
        retcode = sp.Popen(cmd, shell=True).wait()
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            return retcode
    except OSError, e:
        print("Execution failed:", e, file=sys.stderr)

def u_to_ascii(text):
    "Converts given unicode text to ascii with meaningufl translations."
    if isinstance(text, unicode):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
    return text

# Queues: LIFOQueue, FIFOQueue, PriorityQueue

class Queue(deque):
    """Queue is an abstract class/interface. There are three types:
        LIFOQueue(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(lt): Queue where items are sorted by lt, (default <).
    Each type supports the following methods and functions:
        q.put(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.get()         -- return first/last/min/max item from the queue depending on the queue type.
        q.peek()        -- return the element which would be obtaind from q.get() without actually
                           removing it from the queue.
        len(q)          -- number of items in q
        PriorityQueue doesn't extend this class but supports the same interface.
    """
    def __init__(self, seq=None):
        if seq: super(Queue, self).__init__(seq)
        else: super(Queue, self).__init__()

    def put(self, item):
        super(Queue, self).append(item)

    def get(self):
        abstract

    def peek(self):
        abstract

class FIFOQueue(Queue):
    """Insert at the bottom and get elements from the top.
    >>> fq = FIFOQueue()
    >>> fq.put(1); fq.put(2); fq.put(3)
    >>> fq.get()
    1
    >>> fq = FIFOQueue(range(4))
    >>> fq.get()
    0
    >>> print(fq)
    deque([1, 2, 3])
    >>> fq.put(4)
    >>> print(fq)
    deque([1, 2, 3, 4])
    >>> fq.get()
    1
    >>> fq.peek()
    2
    >>> fq.peek()
    2
    >>> fq.get()
    2
    >>> fq.peek()
    3
    """
    def __init__(self, seq=None):
        super(FIFOQueue, self).__init__(seq)
    def get(self):
        return super(FIFOQueue, self).popleft()
    def peek(self):
        return self[0]


class LIFOQueue(Queue):
    """Insert at the bottom and get elements from the bottom.
    >>> lq = LIFOQueue()
    >>> lq.put(1); lq.put(2); lq.put(3)
    >>> print(lq)
    deque([1, 2, 3])
    >>> lq.get()
    3
    >>> lq = LIFOQueue(range(4))
    >>> print(lq)
    deque([0, 1, 2, 3])
    >>> lq.get()
    3
    >>> lq.put(4)
    >>> print(lq)
    deque([0, 1, 2, 4])
    >>> lq.peek()
    4
    >>> lq.get()
    4
    >>> lq.peek()
    2
    >>> lq.get()
    2
    """
    def __init__(self, seq=None):
        super(LIFOQueue, self).__init__(seq)
    def get(self):
        return super(LIFOQueue, self).pop()
    def peek(self):
        return self[-1]


class PriorityQueue(object):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    >>> pq = PriorityQueue()
    >>> pq.put(5); pq.put(1); pq.put(3)
    >>> pq.get()
    1
    >>> pq = PriorityQueue(order=max, seq=[1,9,3,5])
    >>> len(pq)
    4
    >>> pq.get()
    9
    >>> len(pq)
    3
    >>> pq.put(100)
    >>> pq.put(99)
    >>> pq.put(-5)
    >>> pq.get()
    100
    >>> pq.get()
    99
    >>> pq.get()
    5
    >>> pq = PriorityQueue(key=lambda x: x * x, seq = [-3, -2, 1, 5, -10])
    >>> pq.get()
    1
    >>> pq.get()
    -2
    >>> pq.get()
    -3
    >>> pq = PriorityQueue(order=max, key=lambda x: x * x, seq = [-3, -2, 1, 5, -10])
    >>> pq.get()
    -10
    >>> pq.get()
    5
    >>> pq.get()
    -3
    >>> pq.get()
    -2
    >>> pq = PriorityQueue(order=max, key=lambda x: x * x, seq = [-3, -2, 1, 5, -10])
    >>> pq.peek()
    -10
    >>> pq.get()
    -10
    >>> pq.peek()
    5
    >>> pq.get()
    5
    >>> pq = PriorityQueue(range(15))
    >>> pq.nsmallest(5)
    [0, 1, 2, 3, 4]
    >>> pq.nlargest(5)
    [14, 13, 12, 11, 10]
    """
    def __init__(self, seq=None, order=min, key=lambda x: x):
        if order != min: fn = lambda x: -key(x)
        else: fn = key
        if seq:
            seq = [(fn(item), item) for item in seq]
            heapq.heapify(seq)
        update(self, A=(seq or []), order=order, fn=fn)

    def put(self, item):
        heapq.heappush(self.A, (self.fn(item), item))

    def __len__(self):
        return len(self.A)

    def get(self):
        return heapq.heappop(self.A)[1]

    def peek(self):
        return self.A[0][1]

    def extend(self, seq):
        for item in seq:
            heapq.heappush(self.A, (self.fn(item), item))

    def nsmallest(self, n, reverse=False):
        if n > len(self.A):
            items = self.A
        else:
            items = sorted(self.A, reverse=reverse)[:n]
        return map(itemgetter(1), items)

    def nlargest(self, n):
        return self.nsmallest(n, reverse=True)

    def __repr__(self):
        return repr(map(itemgetter(1), self.A))

## {{{ http://code.activestate.com/recipes/576611/ (r11)
class Counter(dict):
    '''Dict subclass for counting hashable objects.  Sometimes called a bag
    or multiset.  Elements are stored as dictionary keys and their counts
    are stored as dictionary values.

    >>> Counter('zyzygy')
    Counter({'y': 3, 'z': 2, 'g': 1})

    '''

    def __init__(self, iterable=None, **kwds):
        '''Create a new, empty Counter object.  And if given, count elements
        from an input iterable.  Or, initialize the count from another mapping
        of elements to their counts.

        >>> c = Counter()                           # a new, empty counter
        >>> c = Counter('gallahad')                 # a new counter from an iterable
        >>> c = Counter({'a': 4, 'b': 2})           # a new counter from a mapping
        >>> c = Counter(a=4, b=2)                   # a new counter from keyword args

        '''
        self.update(iterable, **kwds)

    def __missing__(self, key):
        return 0

    def most_common(self, n=None):
        '''List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.

        >>> Counter('abracadabra').most_common(3)
        [('a', 5), ('r', 2), ('b', 2)]

        '''
        if n is None:
            return sorted(self.iteritems(), key=itemgetter(1), reverse=True)
        return nlargest(n, self.iteritems(), key=itemgetter(1))

    def elements(self):
        '''Iterator over elements repeating each as many times as its count.

        >>> c = Counter('ABCABC')
        >>> sorted(c.elements())
        ['A', 'A', 'B', 'B', 'C', 'C']

        If an element's count has been set to zero or is a negative number,
        elements() will ignore it.

        '''
        for elem, count in self.iteritems():
            for _ in repeat(None, count):
                yield elem

    # Override dict methods where the meaning changes for Counter objects.

    @classmethod
    def fromkeys(cls, iterable, v=None):
        raise NotImplementedError(
            'Counter.fromkeys() is undefined.  Use Counter(iterable) instead.')

    def update(self, iterable=None, **kwds):
        '''Like dict.update() but add counts instead of replacing them.

        Source can be an iterable, a dictionary, or another Counter instance.

        >>> c = Counter('which')
        >>> c.update('witch')           # add elements from another iterable
        >>> d = Counter('watch')
        >>> c.update(d)                 # add elements from another counter
        >>> c['h']                      # four 'h' in which, witch, and watch
        4

        '''
        if iterable is not None:
            if hasattr(iterable, 'iteritems'):
                if self:
                    self_get = self.get
                    for elem, count in iterable.iteritems():
                        self[elem] = self_get(elem, 0) + count
                else:
                    dict.update(self, iterable) # fast path when counter is empty
            else:
                self_get = self.get
                for elem in iterable:
                    self[elem] = self_get(elem, 0) + 1
        if kwds:
            self.update(kwds)

    def copy(self):
        'Like dict.copy() but returns a Counter instance instead of a dict.'
        return Counter(self)

    def __delitem__(self, elem):
        'Like dict.__delitem__() but does not raise KeyError for missing values.'
        if elem in self:
            dict.__delitem__(self, elem)

    def __repr__(self):
        if not self:
            return '%s()' % self.__class__.__name__
        items = ', '.join(map('%r: %r'.__mod__, self.most_common()))
        return '%s({%s})' % (self.__class__.__name__, items)

    # Multiset-style mathematical operations discussed in:
    #       Knuth TAOCP Volume II section 4.6.3 exercise 19
    #       and at http://en.wikipedia.org/wiki/Multiset
    #
    # Outputs guaranteed to only include positive counts.
    #
    # To strip negative and zero counts, add-in an empty counter:
    #       c += Counter()

    def __add__(self, other):
        '''Add counts from two counters.

        >>> Counter('abbb') + Counter('bcc')
        Counter({'b': 4, 'c': 2, 'a': 1})


        '''
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for elem in set(self) | set(other):
            newcount = self[elem] + other[elem]
            if newcount > 0:
                result[elem] = newcount
        return result

    def __sub__(self, other):
        ''' Subtract count, but keep only results with positive counts.

        >>> Counter('abbbc') - Counter('bccd')
        Counter({'b': 2, 'a': 1})

        '''
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for elem in set(self) | set(other):
            newcount = self[elem] - other[elem]
            if newcount > 0:
                result[elem] = newcount
        return result

    def __or__(self, other):
        '''Union is the maximum of value in either of the input counters.

        >>> Counter('abbb') | Counter('bcc')
        Counter({'b': 3, 'c': 2, 'a': 1})

        '''
        if not isinstance(other, Counter):
            return NotImplemented
        _max = max
        result = Counter()
        for elem in set(self) | set(other):
            newcount = _max(self[elem], other[elem])
            if newcount > 0:
                result[elem] = newcount
        return result

    def __and__(self, other):
        ''' Intersection is the minimum of corresponding counts.

        >>> Counter('abbb') & Counter('bcc')
        Counter({'b': 1})

        '''
        if not isinstance(other, Counter):
            return NotImplemented
        _min = min
        result = Counter()
        if len(self) < len(other):
            self, other = other, self
        for elem in ifilter(self.__contains__, other):
            newcount = _min(self[elem], other[elem])
            if newcount > 0:
                result[elem] = newcount
        return result

if __name__ == '__main__':
    import doctest
    doctest.testmod()

