import sys
import datetime
# ensure package import from repo root
sys.path.insert(0, '.')
from easyQuake.easyQuake import detection_continuous

print('Starting EQTransformer inline test')
try:
    # create an explicit project folder for the test so sqlite can create files there
    import os
    proj = os.path.abspath('./test_eqt_project')
    os.makedirs(os.path.join(proj, 'gpd_predict', 'testdata'), exist_ok=True)
    # copy the bundled dayfile into the working test dir if it exists in repo
    try:
        import shutil
        src_dayfile = os.path.join('gpd_predict', 'testdata', 'dayfile.in')
        dst = os.path.join(proj, 'gpd_predict', 'testdata', 'dayfile.in')
        if os.path.exists(src_dayfile) and not os.path.exists(dst):
            shutil.copy(src_dayfile, dst)
    except Exception:
        pass

    detection_continuous(dirname='gpd_predict/testdata', project_folder=proj, project_code='test', local=True, machine=True, machine_picker='EQTransformer', single_date=datetime.date.today(), make3=True)
    print('detection_continuous completed')
except Exception as e:
    import traceback
    print('detection_continuous raised exception:')
    traceback.print_exc()
    raise
