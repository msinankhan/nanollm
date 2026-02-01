import torch
import torch.nn.functional as F
import signal
import warnings

from contextlib import contextmanager
from collections import deque
from nanollm.commons import compute_init, autodetect_device_type
from nanollm.checkpoint_manager import load_model
from contextlib import nullcontext


@contextmanager
def timeout(duration,formula): 
    """
    This wraps execution in a wall-clock timeout enforced by the OS.
    Python’s eval() can hang (e.g. infinite loops, massive computation).
    We want hard time limits.
    """
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration}s")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula,max_time=3):
    try:
        with timeout(max_time,formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",SyntaxWarning)
                return eval(formula, {"__builtins__":{}},{}) # Eval basically takes a string and executes it as a Python expression. Ex.: eval("1 + 2 * 3") = 7
                                                             # eval(expr, globals, locals) is eval's signature. By making globals_dict["__builtins__"]= {}, we are overriding
                                                             #the real python builtins. In this universe: No functions exist; No imports exist ; No IO exists
                                                             # locals = {} → empty namespace

    except Exception as e:
        signal.alarm(0) #Cancels any pending alarm.
        return None
    

def use_calculator(exp):
    exp=exp.replace(",","")

    if all([x in "0123456789*+-/.() " for x in exp]):
        if "**" in exp:
            return None
        return eval_with_timeout(exp)
    
    allowed_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in exp]):
        return None
    
    dangerous_expressions=['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    
    exp_lower=exp.lower()

    if any(pattern in exp_lower for pattern in dangerous_expressions):
        return None
    
    if '.count(' not in exp:
        return None
    
    return eval_with_timeout(exp)

