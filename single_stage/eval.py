import os
import sys
from .config import get_parser

# Thin wrapper: reuse the project-wide eval.py so the same evaluation features apply.
# Single-stage checkpoints contain backbone/offset head params with the same names as PONet,
# so PO_multi_TTA/eval.py can load them (strict=False).

def main():
    cfg = get_parser(); cfg = cfg.parse_args()
    # adapt fields for eval script compatibility
    cfg.task = 'eval'
    # dynamically load top-level eval.py
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    eval_path = os.path.join(root, 'eval.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location('po_eval_single', eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.eval(cfg)

if __name__ == '__main__':
    main()
