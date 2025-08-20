#!/usr/bin/env python3
import sys, os
from datetime import date

# Make sure the workspace package is first on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Preload the local EQTransformer mseed_predictor module and register it under
# the package name so relative imports inside `detection_continuous` will pick
# the workspace module instead of any installed package with the same name.
import importlib.util
local_mseed = os.path.join(os.path.dirname(__file__), 'EQTransformer', 'mseed_predictor.py')
if os.path.exists(local_mseed):
	try:
		spec = importlib.util.spec_from_file_location('easyQuake.EQTransformer.mseed_predictor', local_mseed)
		if spec and spec.loader:
			mod = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(mod)
			sys.modules['easyQuake.EQTransformer.mseed_predictor'] = mod
			# If the EQTransformer conversion produced a .keras model, prefer it for inline loading
			conv = os.path.join(os.path.dirname(local_mseed), 'EqT_model.sanitized.keras')
			if os.path.exists(conv):
				try:
					setattr(mod, '__input_model_override__', conv)
				except Exception:
					pass
	except Exception:
		# ignore preload failures and continue; subsequent code will attempt to import normally
		pass

import importlib
import glob
import types
# Ensure the workspace package is used instead of any installed package:
# create a package module for 'easyQuake' that points to the workspace folder,
# then load `easyQuake.easyQuake` from the local file so relative imports resolve
# to the workspace code (this avoids using an installed easyQuake package).
sys.modules.pop('easyQuake', None)
sys.modules.pop('easyQuake.easyQuake', None)
pkg_mod = types.ModuleType('easyQuake')
pkg_mod.__path__ = [os.path.abspath(os.path.dirname(__file__))]
pkg_mod.__file__ = os.path.join(pkg_mod.__path__[0], '__init__.py')
sys.modules['easyQuake'] = pkg_mod

workspace_easyquake_path = os.path.join(os.path.dirname(__file__), 'easyQuake.py')
spec_eq = importlib.util.spec_from_file_location('easyQuake.easyQuake', workspace_easyquake_path)
workspace_eq = importlib.util.module_from_spec(spec_eq)
# ensure package context is set so relative imports inside easyQuake.py work
workspace_eq.__package__ = 'easyQuake'
spec_eq.loader.exec_module(workspace_eq)
detection_continuous = getattr(workspace_eq, 'detection_continuous')

# Use small test data in gpd_predict/testdata (existing test harness expects this)
proj = 'test_project'

# Provide a concrete date instead of None so detection_continuous can compute start/stop times.
# The test miniSEED files are generic; 2020-01-01 is a safe fixed choice for the demo harness.
single_date = date(2020, 1, 1)

# If a converted .keras model exists in the workspace, export its path so
# the EQTransformer loader will prefer it and avoid legacy HDF5 deserialization.
maybe_keras = os.path.join(os.path.dirname(__file__), 'EQTransformer', 'EqT_model.sanitized.keras')
if os.path.exists(maybe_keras):
	os.environ['EASYQUAKE_EQT_MODEL'] = maybe_keras

# Ensure the project testdata directory exists and contains the sample files from the repo so
# sqlite can create its DB file there (detection_continuous expects files under project_folder/dirname).
import shutil

src_testdata = os.path.join(os.path.dirname(__file__), 'gpd_predict', 'testdata')
dst_testdata = os.path.join(os.getcwd(), proj, 'gpd_predict', 'testdata')
os.makedirs(dst_testdata, exist_ok=True)

# If the user has test mseed files in ~/easyQuake/tests, create a dayfile here first
dayfile_written = False
user_tests = os.path.expanduser('~/easyQuake/tests')
if os.path.isdir(user_tests):
	n = os.path.join(user_tests, 'O2.WILZ.EHN.mseed')
	e = os.path.join(user_tests, 'O2.WILZ.EHE.mseed')
	z = os.path.join(user_tests, 'O2.WILZ.EHZ.mseed')
	if os.path.exists(n) and os.path.exists(e) and os.path.exists(z):
		dayfile_path = os.path.join(dst_testdata, 'dayfile.in')
		with open(dayfile_path, 'w') as dfh:
			dfh.write(f"{n} {e} {z}\n")
		dayfile_written = True
for fname in os.listdir(src_testdata):
	# avoid overwriting a dayfile we just created from the user's tests
	if dayfile_written and fname == 'dayfile.in':
		continue
	srcf = os.path.join(src_testdata, fname)
	dstf = os.path.join(dst_testdata, fname)
	# copy files (overwrite if present)
	try:
		shutil.copy(srcf, dstf)
	except IsADirectoryError:
		pass

# Preload the local EQTransformer mseed_predictor module and register it under
# the package name so relative imports inside `detection_continuous` will pick
# the workspace module instead of any installed package with the same name.
import importlib.util
import types
local_mseed = os.path.join(os.path.dirname(__file__), 'EQTransformer', 'mseed_predictor.py')
if os.path.exists(local_mseed):
	spec = importlib.util.spec_from_file_location('easyQuake.EQTransformer.mseed_predictor', local_mseed)
	if spec and spec.loader:
		mod = importlib.util.module_from_spec(spec)
		try:
			spec.loader.exec_module(mod)
			sys.modules['easyQuake.EQTransformer.mseed_predictor'] = mod
			# If the EQTransformer conversion produced a .keras model, prefer it for inline loading
			conv = os.path.join(os.path.dirname(local_mseed), 'EqT_model.sanitized.keras')
			if os.path.exists(conv):
				try:
					setattr(mod, '__input_model_override__', conv)
				except Exception:
					pass
		except Exception:
			# if loading fails, continue and let detection_continuous attempt inline import
			pass

	# create a dayfile in the project testdata that points to user's ~/easyQuake/tests files if present
	user_tests = os.path.expanduser('~/easyQuake/tests')
	if os.path.isdir(user_tests):
		# try to find O2.WILZ.EH*.mseed
		n = glob.glob(os.path.join(user_tests, 'O2.WILZ.EHN.mseed'))
		e = glob.glob(os.path.join(user_tests, 'O2.WILZ.EHE.mseed'))
		z = glob.glob(os.path.join(user_tests, 'O2.WILZ.EHZ.mseed'))
		if n and e and z:
			dayfile_path = os.path.join(dst_testdata, 'dayfile.in')
			with open(dayfile_path, 'w') as dfh:
				dfh.write(f"{n[0]} {e[0]} {z[0]}\n")

	detection_continuous(dirname='gpd_predict/testdata', project_folder=proj, project_code='test', local=True, machine=True, machine_picker='EQTransformer', single_date=single_date, make3=True)
print('detection_continuous completed')
