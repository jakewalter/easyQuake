"""
Small utility to inspect Keras model JSON or HDF5 for Lambda layers and model_config.
Usage:
  python gpd_predict/inspect_model_h5_json.py model_pol.json
  python gpd_predict/inspect_model_h5_json.py model_pol_legacy_fixed_full.h5

It prints top-level HDF5 keys, attrs, and extracts Lambda layer configs from JSON or model_config.
"""
import sys
import os
import json

def inspect_json(path):
    with open(path,'r') as f:
        j = json.load(f)
    # JSON may be model or models; handle common shapes
    def find_lambdas(obj, prefix=''):
        if isinstance(obj, dict):
            if obj.get('class_name') == 'Lambda' or obj.get('class_name') == 'TimeDistributed' and isinstance(obj.get('config'), dict) and obj['config'].get('layer',{}).get('class_name')=='Lambda':
                print(prefix + 'Found Lambda layer:', obj.get('config',{}).get('name', '<unknown>'))
                print('  config keys:', list(obj.get('config',{}).keys()))
                # print function config if present
                cfg = obj.get('config',{})
                if 'function' in cfg:
                    print('  function (truncated):')
                    s = str(cfg['function'])
                    print('   ', s[:1000])
                if 'callable' in cfg:
                    print('  callable (truncated):')
                    print('   ', str(cfg['callable'])[:1000])
            for k,v in obj.items():
                find_lambdas(v, prefix + '/' + str(k))
        elif isinstance(obj, list):
            for i,v in enumerate(obj):
                find_lambdas(v, prefix + f'[{i}]')
    find_lambdas(j)

def inspect_h5(path):
    try:
        import h5py
    except Exception as e:
        print('h5py not available:', e)
        return
    print('HDF5 size:', os.path.getsize(path))
    with h5py.File(path,'r') as f:
        print('Top keys:', list(f.keys()))
        # try attrs
        for k,v in f.attrs.items():
            if k == 'model_config' or k == 'keras_version':
                print('attr',k,':', str(v)[:1000])
        # model_config dataset
        if 'model_config' in f:
            try:
                mc = f['model_config'][()]
                if isinstance(mc, bytes):
                    mc = mc.decode('utf-8', errors='replace')
                print('\nmodel_config (truncated):')
                print(mc[:2000])
                try:
                    j = json.loads(mc)
                    print('\n-- scanning embedded JSON for Lambda layers --')
                    inspect_json_obj = j
                    def find_lambdas_obj(obj, prefix=''):
                        if isinstance(obj, dict):
                            if obj.get('class_name') == 'Lambda':
                                print(prefix + 'Found Lambda layer:', obj.get('config',{}).get('name', '<unknown>'))
                                cfg = obj.get('config',{})
                                if 'function' in cfg:
                                    print('  function (truncated):')
                                    s = str(cfg['function'])
                                    print('   ', s[:1000])
                            for k,v in obj.items():
                                find_lambdas_obj(v, prefix + '/' + str(k))
                        elif isinstance(obj, list):
                            for i,v in enumerate(obj):
                                find_lambdas_obj(v, prefix + f'[{i}]')
                    find_lambdas_obj(inspect_json_obj)
                except Exception as e:
                    print('Could not parse model_config JSON:', e)
            except Exception as e:
                print('Could not read model_config dataset:', e)
        else:
            print('No "model_config" dataset in HDF5; model may be weights-only.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python gpd_predict/inspect_model_h5_json.py <model.json|model.h5>')
        sys.exit(2)
    p = sys.argv[1]
    if not os.path.exists(p):
        print('File not found:', p)
        sys.exit(2)
    if p.lower().endswith('.json'):
        print('Inspecting JSON:', p)
        try:
            inspect_json(p)
        except Exception as e:
            print('Error inspecting JSON:', e)
    else:
        print('Inspecting HDF5:', p)
        inspect_h5(p)
