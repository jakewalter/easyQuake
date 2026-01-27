import os
import time
import threading
import queue
import logging
import subprocess
import glob
from obspy import UTCDateTime
from easyQuake import detection_association_event

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def watch_for_rtxml_and_run(project_folder, picker):
    processed = set()
    last_full_scan = time.time()
    scan_interval = 3600  # 1 hour in seconds
    while True:
        now = time.time()
        # Always scan for new rt.xml files in subfolders
        for rootdir, dirs, files in os.walk(project_folder):
            # Skip the root folder itself
            if os.path.abspath(rootdir) == os.path.abspath(project_folder):
                continue
            for fname in files:
                if fname == "rt.xml":
                    fullpath = os.path.join(rootdir, fname)
                    # Check for any file starting with '1dass' in this subfolder
                    has_1dass = any(f.startswith("1dass") for f in files)
                    if fullpath not in processed and not has_1dass:
                        logger.info(f"Detected new rt.xml: {fullpath}")
                        stalist = None
                        subdir = fullpath.split('/')[-2]
                        print(subdir)
                        threading.Thread(
                            target=run_detection_threading_async,
                            args=(stalist, project_folder, subdir, picker),
                            daemon=True
                        ).start()
                        processed.add(fullpath)
        # Every hour, re-check all subfolders for missed rt.xml files that should be in the queue
        if now - last_full_scan > scan_interval:
            logger.info("Performing periodic full scan for rt.xml files needing to be queued...")
            for rootdir, dirs, files in os.walk(project_folder):
                if os.path.abspath(rootdir) == os.path.abspath(project_folder):
                    continue
                for fname in files:
                    if fname == "rt.xml":
                        fullpath = os.path.join(rootdir, fname)
                        has_1dass = any(f.startswith("1dass") for f in files)
                        if fullpath not in processed and not has_1dass:
                            logger.info(f"Periodic scan: Detected rt.xml needing to be queued: {fullpath}")
                            stalist = None
                            subdir = fullpath.split('/')[-2]
                            print(subdir)
                            threading.Thread(
                                target=run_detection_threading_async,
                                args=(stalist, project_folder, subdir, picker),
                                daemon=True
                            ).start()
                            processed.add(fullpath)
            last_full_scan = now
        time.sleep(5)

def watch_for_seiscomp_and_scp(project_folder, remote_ip, remote_path, remote_user):
    processed = set()
    try:
        while True:
            files = glob.glob(os.path.join(project_folder, "*_seiscomp.xml"))
                # Do NOT skip the root directory, since _seiscomp.xml files are in the root
            for fname in files:
                #logger.debug(f"Found file: {fname}")
                if fname.endswith("_seiscomp.xml"):
                    fullpath = fname
                    if fullpath not in processed:
                        logger.info(f"Detected seiscomp file: {fullpath}")
                        try:
                            remote = f"{remote_user}@{remote_ip}:{remote_path}"
                            logger.info(f"Copying {fullpath} to {remote} (passwordless scp)")
                            subprocess.run(
                                ["scp", fullpath, remote],
                                check=True
                            )
                            logger.info(f"Successfully copied {fullpath} to {remote}")
                        except Exception as exc:
                            logger.error(f"SCP failed for {fullpath}: {exc}")
                        processed.add(fullpath)
            time.sleep(5)
    except Exception as e:
        logger.error(f"Exception in watch_for_seiscomp_and_scp: {e}")

def run_detection_threading_async(stalist, project_folder, subdir, picker):
    try:
        while True:
            logger.info(f"Triggering detection for {subdir}")
            
            # Verify the target directory exists and contains data
            target_dir = os.path.join(project_folder, subdir)
            if not os.path.exists(target_dir):
                logger.error(f"Target directory does not exist: {target_dir}")
                break
                
            # Check for data files
            data_files = glob.glob(os.path.join(target_dir, "*.mseed"))
            if not data_files:
                logger.warning(f"No mseed files found in {target_dir}")
                break
                
            logger.info(f"Found {len(data_files)} data files in {target_dir}")
            
            try:
                # Convert subdir timestamp to ISO format
                if len(subdir) == 14:  # YYYYMMDDHHMMSS
                    approx_time = f"{subdir[0:4]}-{subdir[4:6]}-{subdir[6:8]}T{subdir[8:10]}:{subdir[10:12]}:{subdir[12:14]}"
                else:
                    from obspy import UTCDateTime
                    approx_time = UTCDateTime().isoformat()
                
                logger.info(f"Processing {subdir} with timestamp: {approx_time}")
                
                # Run detection with correct parameters - use project_folder as base directory
                # The function will construct the subdirectory path from approx_time
                cmd = [
                    "python",
                    "-c",
                    (
                        "from easyQuake import detection_association_event;"
                        "detection_association_event("
                        f"'{project_folder}', 'ok', 300, 300, True, True, '{picker}', "
                        "'/home/jwalter/anaconda3/envs/easyquake/bin/python', "
                        f"'{approx_time}', False)"
                    )
                ]
                
                logger.debug(f"Executing command: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Log any output for debugging
                if result.stdout:
                    logger.info(f"Detection output for {subdir}: {result.stdout}")
                if result.stderr:
                    logger.warning(f"Detection stderr for {subdir}: {result.stderr}")
                
                logger.info(f"Successfully completed detection for {subdir}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"detection_association_event failed for {subdir}: {e}")
                if e.stdout:
                    logger.error(f"stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"stderr: {e.stderr}")
            except Exception as exc:
                logger.error(f"Unexpected error for {subdir}: {exc}")
            
            break  # Only run once per thread
            
    except KeyboardInterrupt:
        logger.info("Stopping async data grabber...")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        project_folder = sys.argv[1]
        picker = sys.argv[2]
    else:
        project_folder = '/home/jwalter/scratch'
        picker = 'PhaseNet'  # <-- set your picker here (e.g., 'PhaseNet', 'EQTransformer', etc.)
    logger.info(f"Watching for rt.xml in subfolders of {project_folder} using picker {picker}")

    # Hardcoded server info for passwordless scp
    remote_ip = "192.168.1.100"  # <-- set your server IP here
    remote_user = "remoteuser"   # <-- set your server username here
    remote_path = "/home/remoteuser/seiscomp_uploads/"  # <-- set your server path here

    # Start both listeners in separate threads
    threading.Thread(
        target=watch_for_rtxml_and_run,
        args=(project_folder, picker),
        daemon=True
    ).start()
    threading.Thread(
        target=watch_for_seiscomp_and_scp,
        args=(project_folder, remote_ip, remote_path, remote_user),
        daemon=True
    ).start()

    # Keep main thread alive
    while True:
        time.sleep(60)
