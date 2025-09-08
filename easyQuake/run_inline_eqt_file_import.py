#!/usr/bin/env python3
import importlib.util, importlib.machinery, os, sys
# load the easyQuake.py file as a module without triggering package-level imports
file_path = os.path.join(os.path.dirname(__file__), 'easyQuake.py')
spec = importlib.util.spec_from_loader('easyquake_file', importlib.machinery.SourceFileLoader('easyquake_file', file_path))
emod = importlib.util.module_from_spec(spec)
# execute module in isolation
spec.loader.exec_module(emod)

# call detection_continuous if present
if hasattr(emod, 'detection_continuous'):
    emod.detection_continuous(dirname='gpd_predict/testdata', project_folder='test_project', project_code='test', local=True, machine=True, machine_picker='EQTransformer', single_date=None, make3=True)
    print('detection_continuous completed')
else:
    print('detection_continuous not found in easyQuake.py')
