# easyQuake (realtime) — README

## Overview
Simple instructions to run the realtime scripts in this folder on a server. These scripts typically need small parameter edits (paths, host/port, model names) for your environment.

---

## Files
- `rt_easyquake.py` — Realtime processing script. Adjust data paths and model names inside this file or pass them as CLI args if supported.
- `seedlink_connection_v5.py` — SeedLink connection client. Edit host/port, station list, and output directories to match your SeedLink server.
- `seedlink_sds_connection.py` — Alternate SeedLink->SDS client. Also requires server and output path customization.

---

## Quick start (run on GPU/easyQuake server) 🔧

1. Run the main scripts in the background using `nohup` (examples):

```bash
# Start realtime processing (edit folder and model name to your environment)
nohup python rt_easyquake.py /scratch/realtime PhaseNet &

# Start seedlink client
nohup python seedlink_connection_v5.py &
```

Notes:
- Replace `/scratch/realtime` and `PhaseNet` with the actual data directory and model name used on your server.
- `nohup` writes output to `nohup.out` by default. Use `tail -f nohup.out` to watch logs.
- Use `&` to run in background. To stop: `ps aux | grep rt_easyquake` and `kill <PID>` or use systemd/supervisor for managed services.

---

## Edit parameters (what to check)
Open each script and look near the top for constants or default variables. Common items to update:
- `DATA_DIR` / path arguments (e.g. `/scratch/realtime`)
- `MODEL_NAME` or model path (e.g. `PhaseNet`)
- `SEEDLINK_HOST`, `SEEDLINK_PORT`
- `OUTPUT_DIR` or SDS path
- Logging level and file paths

If a script accepts CLI args, prefer passing values on the command line so you don't have to edit the file.

---

## Running as a service (recommended for production) ⚙️
Use `systemd`, `supervisord`, or similar to keep the process running and to capture logs. Example minimal `systemd` unit for `rt_easyquake`:

```ini
[Unit]
Description=easyQuake realtime
After=network.target

[Service]
User=youruser
WorkingDirectory=/path/to/realtime
ExecStart=/usr/bin/python /path/to/realtime/rt_easyquake.py /scratch/realtime PhaseNet
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Troubleshooting & tips
- Check `nohup.out` or configured log files for errors.
- Confirm Python version and dependencies.
- Ensure file and directory permissions allow read/write where needed.
- If connecting to SeedLink, verify server connectivity (telnet/openssl) and ports.

---

## SeisComP ingestion (ingest to remote SeisComP server) 🛰️
If you have a SeisComP instance running on another server, the easiest approach is to run an inotify-based dispatcher that watches the directory where your scripts write XML event messages and calls `scdispatch` when new XML files appear.

**Steps:**

1. Install prerequisites on the SeisComP server:

```bash
sudo apt-get install inotify-tools  # or the equivalent for your distro
```

2. Create the `inotify2.bash` script (example):

```bash
#!/bin/bash
set -x # This will print each command as it's executed

/usr/bin/inotifywait -m /home/sysop/incoming_ML -e create -e moved_to |
    while read directory action file; do
        echo "Detected event: Directory: $directory, Action: $action, File: $file"
        if [[ "$file" =~ .*xml$ ]]; then # Does the file end with .xml?
            echo "File $file matches .xml pattern. Attempting to dispatch."
            scdispatch -i "$directory$file" -O add # Use "$directory/$file" for full path safety
            dispatch_status=$? # Capture the exit status of scdispatch
            echo "scdispatch exited with status: $dispatch_status"

            # You might want to check the status before removing
            # if [ $dispatch_status -eq 0 ]; then
                echo "Removing file: $directory$file"
                rm "$directory$file"
            # else
            #    echo "scdispatch failed for $file, not removing."
            # fi
        else
            echo "File $file does not match .xml pattern. Skipping."
        fi
    done
```

3. Make the script executable and run it (as `sysop` or appropriate user), or create a `systemd` unit/supervisor job to run it persistently:

```bash
chmod +x /path/to/inotify2.bash
nohup /path/to/inotify2.bash &  # or use systemd for production
```

**Notes:**
- Adjust `/home/sysop/incoming_ML` to the directory where your XML files are written.
- Ensure `scdispatch` is in PATH for the script's user (or use the full path to `scdispatch`).
- If your production setup has the XML files on a different machine, either run this script there or copy files securely (rsync/ssh) to the SeisComP server and watch the incoming directory.
- Test with a sample XML first and check logs for successful dispatch.

---

If you want, I can add a `systemd` unit for `inotify2.bash` or provide an SSH-based example to dispatch files remotely. 

---

## Keep the SeisComP listener running (cron) 
If you'd like the `inotify2.bash` listener to be restarted automatically by cron, add the following cron entry on the SeisComP server (checks to make sure that it is running every 2 minutes, and restarts if it is not):

```cron
*/2 * * * * /usr/bin/bash /home/sysop/bin/cron_restart_ml_listen.bash >/dev/null 2>&1
```

Create `/home/sysop/bin/cron_restart_ml_listen.bash` with the following content and make it executable (`chmod +x`):

```bash
#!/bin/bash
#
MAILTO=""
PROCESS='inotifywait'
if ps ax | grep -v grep | grep $PROCESS > /dev/null
then
    echo "Seiscomp Listener running, EXIT"
    exit
else
    echo "$PROCESS is not running"
    echo "start the process"
    echo "Start $PROCESS !"
    #echo "put in the start command here"
    /home/sysop/bin/start_MLlisten.bash &
fi
```

And create `/home/sysop/bin/start_MLlisten.bash` to set up the environment and start the listener (make executable):

```bash
#!/bin/bash
PATH=/home/sysop/bin:$PATH
cd /home/sysop/bin
./inotify2.bash >> ML_listen.log 2>&1 &
#/usr/bin/nohup python seiscomp_processing.py &
```

Notes & tips:
- Adjust the paths and user (`sysop`) as needed for your environment.
- Ensure `inotifywait` and `scdispatch` are installed and available in the PATH for the script's user.
- Test the scripts manually before relying on cron: `/home/sysop/bin/start_MLlisten.bash` and then check `ML_listen.log` and `ps aux | grep inotifywait`.
- To install the cron entry for the `sysop` user, run `crontab -e` as that user and paste the cron line above.


