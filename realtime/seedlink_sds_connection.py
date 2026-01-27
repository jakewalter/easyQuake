#!/usr/bin/env python3
"""
SeedLink Connection Script - Fixed Window Processing
======================================================

This script establishes a continuous connection to SeedLink servers and downloads
seismic data using a buffer-based approach with fixed-length time windows.

Key Features:
- Buffer-based data collection instead of real-time processing
- Fixed-length time windows ensuring all traces have identical start times and durations
- Automatic station discovery from SeedLink servers
- Multi-threaded operation for concurrent data handling
- Quality control with spike detection and data validation
- Automatic cleanup of old data files
- Thread-safe buffer management

This version (v4) replaces the real-time trimming approach of v3 with a buffering
system that extracts perfectly aligned time windows, ensuring all output traces
have exactly the same length and timing.

Dependencies:
- obspy
- numpy
- threading
- collections

Author: Jake Walter
Date: 2024
"""

import os
import time
import threading
import numpy as np
from obspy import Stream, UTCDateTime, read, Trace
from obspy.clients.seedlink import SLClient
from obspy.clients.filesystem.sds import Client as SDSClient
import logging
import yaml
import fnmatch
import glob
import shutil
from collections import defaultdict
import xml.etree.ElementTree as ET
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Silence extremely verbose/info messages from obspy's seedlink connection (same upstream bug)
logging.getLogger('obspy.clients.seedlink.client.seedlinkconnection').setLevel(logging.WARNING)

# Simple helper for rate-limited logging to avoid repeating identical messages
_last_log_time = {}
def rate_limited_log(level, key, interval, msg, *args, **kwargs):
    now = time.time()
    last = _last_log_time.get(key)
    if last is not None and (now - last) < interval:
        return
    _last_log_time[key] = now
    if level == 'warning':
        logger.warning(msg, *args, **kwargs)
    elif level == 'info':
        logger.info(msg, *args, **kwargs)
    elif level == 'debug':
        logger.debug(msg, *args, **kwargs)
    elif level == 'error':
        logger.error(msg, *args, **kwargs)
    else:
        logger.log(msg, *args, **kwargs)

# Alerting helpers (webhook / slack / email / callback)
_last_alert_time = {}
def _send_alert_via_webhook(url, payload):
    try:
        import requests
    except Exception:
        logger.debug("requests not available; webhook alert skipped")
        return False
    try:
        requests.post(url, json=payload, timeout=5)
        return True
    except Exception as e:
        logger.debug("Webhook alert failed: %s", e)
        return False

def _send_alert_via_slack(url, subject, message):
    try:
        import requests
    except Exception:
        logger.debug("requests not available; slack alert skipped")
        return False
    try:
        text = f"*{subject}*\n{message}"
        payload = {"text": text}
        requests.post(url, json=payload, timeout=5)
        return True
    except Exception as e:
        logger.debug("Slack webhook alert failed: %s", e)
        return False

def _send_alert_via_email(cfg, subject, message):
    try:
        import smtplib
        from email.message import EmailMessage
    except Exception:
        logger.debug("smtplib not available; email alert skipped")
        return False
    try:
        msg = EmailMessage()
        msg['From'] = cfg.get('from')
        msg['To'] = cfg.get('to')
        msg['Subject'] = subject
        msg.set_content(message)

        host = cfg.get('host', 'localhost')
        port = cfg.get('port', 25)
        user = cfg.get('user')
        password = cfg.get('password')
        use_ssl = cfg.get('ssl', False)

        if use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=10)
        else:
            server = smtplib.SMTP(host, port, timeout=10)
            if cfg.get('starttls', False):
                server.starttls()

        if user:
            server.login(user, password)

        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        logger.debug("Email alert failed: %s", e)
        return False

def send_alert(msg_text, subject=None, key=None, interval=300, alert_callback=None, alert_webhook=None, alert_email=None, alert_slack_url=None):
    if key is not None:
        last = _last_alert_time.get(key)
        now = time.time()
        if last is not None and (now - last) < interval:
            logger.debug("Alert suppressed (rate limit) for key %s", key)
            return False
        _last_alert_time[key] = now

    if subject is None:
        subject = "SeedLink alert"

    if alert_callback is not None and callable(alert_callback):
        try:
            alert_callback(subject, msg_text)
            return True
        except Exception as e:
            logger.debug("alert_callback failed: %s", e)

    if alert_slack_url:
        if _send_alert_via_slack(alert_slack_url, subject, msg_text):
            return True

    if alert_webhook:
        payload = {'subject': subject, 'message': msg_text}
        if _send_alert_via_webhook(alert_webhook, payload):
            return True

    if alert_email and isinstance(alert_email, dict):
        if _send_alert_via_email(alert_email, subject, msg_text):
            return True

    logger.warning("ALERT: %s - %s", subject, msg_text)
    return True

class WaveformBuffer(SLClient):
    """
    SeedLink SLClient subclass that buffers incoming waveforms for fixed-window processing.
    """
    def __init__(self, data_buffer, lock=None):
        super().__init__(loglevel="NOTSET")
        self.data_buffer = data_buffer  # Dictionary to store traces by ID
        self.lock = lock
        self.packet_count = 0

    def packet_handler(self, count, slpack):
        from obspy.clients.seedlink.slpacket import SLPacket
        if slpack is None or (slpack == SLPacket.SLNOPACKET) or (slpack == SLPacket.SLERROR):
            return False
        
        trace = slpack.get_trace()
        if trace is None:
            return False
            
        # Store in buffer by trace ID
        trace_id = trace.id
        with self.lock:
            if trace_id not in self.data_buffer:
                self.data_buffer[trace_id] = Stream()
            
            self.data_buffer[trace_id] += trace
            # Merge to avoid fragmentation but don't trim yet
            if len(self.data_buffer[trace_id]) > 1:
                self.data_buffer[trace_id].merge(-1)
            
            # Clean processing info
            for tr in self.data_buffer[trace_id]:
                tr.stats.processing = []
                
            self.packet_count += 1
            
        return False

def extract_fixed_windows(data_buffer, window_start, window_length, lock):
    """
    Extract fixed-length windows from the data buffer.
    All traces will have exactly the same start time and duration.
    
    Parameters:
    -----------
    data_buffer : dict
        Dictionary of Stream objects keyed by trace ID
    window_start : UTCDateTime
        Exact start time for the window
    window_length : int
        Duration of window in seconds
    lock : threading.Lock
        Lock for thread-safe access to buffer
        
    Returns:
    --------
    list : List of traces with exactly the same time window
    """
    window_end = window_start + window_length
    fixed_traces = []
    debug_info = []
    
    with lock:
        for trace_id, stream in data_buffer.items():
            if len(stream) == 0:
                continue
                
            # Get the trace (should be merged to single trace)
            if len(stream) > 1:
                stream.merge(-1)
                
            trace = stream[0]
            
            # Debug: collect timing info
            debug_info.append({
                'id': trace_id,
                'start': trace.stats.starttime,
                'end': trace.stats.endtime,
                'duration': trace.stats.endtime - trace.stats.starttime
            })
            
            # More flexible window coverage check
            # Allow traces that have ANY overlap with our window and sufficient data
            trace_start = trace.stats.starttime
            trace_end = trace.stats.endtime
            
            # Check for overlap and sufficient coverage
            has_overlap = not (trace_end <= window_start or trace_start >= window_end)
            covers_start = trace_start <= window_start + 5  # Allow 5 second tolerance
            covers_end = trace_end >= window_end - 5  # Allow 5 second tolerance
            
            if has_overlap and covers_start and covers_end:
                # Create a copy and trim to exact window
                fixed_trace = trace.copy()
                fixed_trace.trim(starttime=window_start, endtime=window_end, pad=True, fill_value=0)
                
                # Ensure exact length by resampling if needed
                expected_samples = int(window_length * trace.stats.sampling_rate)
                if len(fixed_trace.data) != expected_samples:
                    # Interpolate to exact length if close
                    if abs(len(fixed_trace.data) - expected_samples) <= 2:
                        if len(fixed_trace.data) < expected_samples:
                            # Pad with zeros
                            padding = expected_samples - len(fixed_trace.data)
                            fixed_trace.data = np.concatenate([fixed_trace.data, np.zeros(padding)])
                        else:
                            # Truncate
                            fixed_trace.data = fixed_trace.data[:expected_samples]
                    else:
                        logger.warning(f"Large sample count mismatch for {trace_id}: got {len(fixed_trace.data)}, expected {expected_samples}")
                        continue
                
                # Set exact timing
                fixed_trace.stats.starttime = window_start
                fixed_trace.stats.npts = len(fixed_trace.data)
                
                fixed_traces.append(fixed_trace)
                logger.debug(f"Created fixed window for {trace_id}: {len(fixed_trace.data)} samples, {fixed_trace.stats.starttime} to {fixed_trace.stats.endtime}")
            else:
                logger.debug(f"Insufficient data for {trace_id}: trace covers {trace.stats.starttime} to {trace.stats.endtime}, need {window_start} to {window_end}")
    
    # Log debug info about trace timing
    if len(debug_info) > 0 and len(fixed_traces) == 0:
        logger.warning(f"Window extraction failed for {window_start} to {window_end}")
        logger.warning(f"Sample trace timings:")
        for i, info in enumerate(debug_info[:5]):  # Show first 5 traces
            logger.warning(f"  {info['id']}: {info['start']} to {info['end']} (duration: {info['duration']:.1f}s)")
    
    return fixed_traces

def cleanup_buffer(data_buffer, window_start, max_buffer_length, lock):
    """
    Clean up old data from buffer to prevent memory issues.
    Keep only data that might be needed for future windows.
    
    Parameters:
    -----------
    data_buffer : dict
        Dictionary of Stream objects keyed by trace ID
    window_start : UTCDateTime
        Current window start time
    max_buffer_length : int
        Maximum time in seconds to keep in buffer
    lock : threading.Lock
        Lock for thread-safe access
    """
    cutoff_time = window_start - max_buffer_length
    
    with lock:
        for trace_id, stream in list(data_buffer.items()):
            if len(stream) == 0:
                continue
                
            # Trim old data from the stream
            try:
                stream.trim(starttime=cutoff_time)
                if len(stream) == 0 or len(stream[0].data) == 0:
                    del data_buffer[trace_id]
                    logger.debug(f"Removed empty trace {trace_id} from buffer")
            except Exception as e:
                logger.warning(f"Error cleaning buffer for {trace_id}: {e}")

def continuous_fixed_window_download(
    server='rtserve.ou.edu:18000',
    stalist=None,
    project_folder='/scratch/realtime',
    window_length=180,
    window_overlap=30,
    buffer_length=1800,  # Keep 30 minutes of data in buffer
    spike_percent=0.95,
    spike_multiplier=2,
    spike_threshold=5,
    # watchdog & alerting options
    stall_threshold=180,
    dedupe_interval=60,
    restart_backoff_initial=5,
    restart_backoff_max=300,
    alert_callback=None,
    alert_webhook=None,
    alert_email=None,
    alert_slack_url=None
):
    """
    Maintain a continuous connection to SeedLink, saving fixed-length time windows.
    All traces in each window will have exactly the same start time and duration.
    Windows start on minute boundaries with configurable overlap.
    
    Parameters:
    -----------
    server : str
        SeedLink server address (default: 'rtserve.ou.edu:18000')
    stalist : list
        List of station selectors in format 'NET_STA:CHAN'
    project_folder : str
        Directory to save waveform data (default: '/scratch/realtime')
    window_length : int
        Time window in seconds for saving data (default: 180 = 3 minutes)
    window_overlap : int
        Overlap in seconds between consecutive windows (default: 30 seconds)
        Each window starts on the minute, but includes overlap with previous window
    buffer_length : int
        How much data to keep in memory buffer (default: 1800 seconds)
    spike_percent : float
        Percentile threshold for spike detection baseline (default: 0.95)
    spike_multiplier : float
        Multiplier for spike detection threshold (default: 2)
    spike_threshold : int
        Maximum allowed number of spikes before rejecting trace (default: 5)
    stall_threshold : int
        Seconds without new packets before treating the stream as stalled (default: 180)
    dedupe_interval : int
        Seconds to suppress duplicate identical log messages sent as alerts (default: 60)
    restart_backoff_initial : int
        Initial backoff seconds for restarting the SeedLink client (default: 5)
    restart_backoff_max : int
        Maximum backoff seconds between restart attempts (default: 300)
    alert_callback, alert_webhook, alert_email, alert_slack_url:
        Optional alert sinks (callback subject,msg) or webhook, email dict, or slack webhook URL.
    """
    lock = threading.Lock()
    data_buffer = defaultdict(Stream)  # Dictionary to store traces by ID
    
    client = WaveformBuffer(data_buffer, lock=lock)
    client.slconn.set_sl_address(server)
    
    # Set up station selection using multiselect
    if stalist and len(stalist) > 0:
        stas = ','.join(stalist)
        logger.info(f"Setting up station selection for {len(stalist)} stations")
        logger.debug(f"Station list: {stas[:200]}...")  # Show first 200 chars
    else:
        stas = '*'
        logger.info("Using wildcard selection for all stations")
    
    client.multiselect = stas
    # reduce obspy SeedLink client verbosity to avoid noisy/malformed INFO messages
    client.verbose = 0
    client.initialize()

    # Start the SeedLink client in a thread
    def run_client():
        try:
            client.run()
        except Exception as e:
            logger.error(f"SeedLink client.run() error: {e}")

    client_thread = threading.Thread(target=run_client, daemon=True)
    client_thread.start()
    
    # Wait for initial data to accumulate (scaled with window size)
    initial_wait = max(60, 2 * window_length + 30)
    logger.info(f"Waiting for initial data accumulation ({initial_wait} seconds)...")
    time.sleep(initial_wait)
    
    # Check if we're getting any data
    with lock:
        buffer_count = len(data_buffer)
        packet_count = client.packet_count
    logger.info(f"After initial wait: {buffer_count} traces in buffer, {packet_count} packets received")
    
    if buffer_count == 0:
        logger.warning("No data received after initial wait. Continuing anyway...")
        send_alert(
            f"No data received from {server} after initial wait ({initial_wait}s)",
            subject="SeedLink: no initial data",
            key=f"initial_no_data_{server}",
            interval=dedupe_interval,
            alert_callback=alert_callback,
            alert_webhook=alert_webhook,
            alert_email=alert_email,
            alert_slack_url=alert_slack_url
        )
    
    # Calculate aligned window start times (aligned to 60 second boundaries = on the minute)
    current_time = UTCDateTime.now()
    # Align to minute boundaries
    minute_boundary = 60
    # Start with a window that's already past to ensure we have data
    # Go back enough to ensure data availability
    aligned_start = UTCDateTime(int((current_time.timestamp - 2 * window_length) // minute_boundary) * minute_boundary)
    last_window_start = aligned_start
    
    # Calculate the step between windows (window length minus overlap)
    window_step = window_length - window_overlap
    
    logger.info(f"Starting fixed window processing at {aligned_start} (current time: {current_time})")
    logger.info(f"Window configuration: {window_length}s windows with {window_overlap}s overlap, stepping every {window_step}s")
    
    # Track retry attempts for each window to avoid infinite loops
    window_retry_count = {}
    max_retries = 10  # Maximum retries before giving up on a window

    # Watchdog / progress tracking for stalled SeedLink clients
    last_packet_count = client.packet_count
    last_progress_time = time.time()
    last_restart_time = 0
    current_backoff = restart_backoff_initial

    try:
        while True:
            current_time = UTCDateTime.now()
            # Next window starts one step forward (not full window length)
            next_window_start = last_window_start + window_step
            
            # Process windows that are sufficiently old (have complete data)
            # Only wait if we're caught up to near real-time
            min_window_age = 60
            window_end_time = next_window_start + window_length
            window_age = current_time - window_end_time
            
            if window_age < min_window_age:
                sleep_time = min_window_age - window_age
                logger.info(f"Caught up to real-time. Waiting {sleep_time:.1f}s for window {next_window_start} to age sufficiently")
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Check if SeedLink client thread is still alive, restart if needed
            if not client_thread.is_alive():
                logger.warning("SeedLink client thread died, restarting...")
                send_alert(
                    f"SeedLink client thread for {server} died and is being restarted",
                    subject="SeedLink thread died",
                    key=f"thread_died_{server}",
                    interval=dedupe_interval,
                    alert_callback=alert_callback,
                    alert_webhook=alert_webhook,
                    alert_email=alert_email,
                    alert_slack_url=alert_slack_url
                )
                client_thread = threading.Thread(target=run_client, daemon=True)
                client_thread.start()
                time.sleep(5)  # Give it a moment to start

            # Watchdog: ensure packets continue to arrive. Restart the
            # client thread if no new packets arrive for `stall_threshold` seconds
            try:
                if client.packet_count > last_packet_count:
                    last_packet_count = client.packet_count
                    last_progress_time = time.time()
                    current_backoff = restart_backoff_initial
                elif time.time() - last_progress_time > stall_threshold:
                    # Only perform a restart if backoff window has passed
                    if time.time() - last_restart_time >= current_backoff:
                        logger.warning("No new packets for %ds; restarting SeedLink client (backoff %ds)", stall_threshold, current_backoff)
                        send_alert(
                            f"No new packets from {server} for {stall_threshold}s; restarting client (backoff {current_backoff}s)",
                            subject="SeedLink watchdog restart",
                            key=f"watchdog_restart_{server}",
                            interval=dedupe_interval,
                            alert_callback=alert_callback,
                            alert_webhook=alert_webhook,
                            alert_email=alert_email,
                            alert_slack_url=alert_slack_url
                        )
                        try:
                            client.slconn.close()
                        except Exception:
                            pass
                        client_thread = threading.Thread(target=run_client, daemon=True)
                        client_thread.start()
                        last_restart_time = time.time()
                        current_backoff = min(current_backoff * 2, restart_backoff_max)
                        # Reset progress tracking after restart
                        last_progress_time = time.time()
                        last_packet_count = client.packet_count
            except Exception:
                logger.debug("Watchdog check failed", exc_info=True)
            
            # Extract fixed windows from buffer
            window_start = next_window_start
            
            # Debug: Check buffer status before extraction
            with lock:
                buffer_traces = len(data_buffer)
                total_packets = client.packet_count
                trace_info = []
                for trace_id, stream in data_buffer.items():
                    if len(stream) > 0:
                        trace = stream[0]
                        trace_info.append(f"{trace_id}: {trace.stats.starttime} to {trace.stats.endtime}")
                        
            logger.info(f"Before extraction for window {window_start}: {buffer_traces} traces, {total_packets} total packets")
            if len(trace_info) > 0:
                logger.debug(f"Sample trace times: {trace_info[:3]}")  # Show first 3 traces
            
            fixed_traces = extract_fixed_windows(data_buffer, window_start, window_length, lock)
            
            if len(fixed_traces) == 0:
                # Track retry attempts for this window
                window_key = str(window_start)
                window_retry_count[window_key] = window_retry_count.get(window_key, 0) + 1

                if window_retry_count[window_key] >= max_retries:
                    logger.error(f"Giving up on window {window_start} after {max_retries} retries - no data available")
                    send_alert(
                        f"Giving up on window {window_start} after {max_retries} retries - no data available for server {server}",
                        subject="SeedLink window give-up",
                        key=f"giveup_{server}_{window_key}",
                        interval=dedupe_interval,
                        alert_callback=alert_callback,
                        alert_webhook=alert_webhook,
                        alert_email=alert_email,
                        alert_slack_url=alert_slack_url
                    )
                    # Move to next window to avoid being stuck forever
                    last_window_start = window_start
                    # Clean up retry counter
                    del window_retry_count[window_key]
                    continue

                rate_limited_log('warning', f'no_complete_{window_key}', dedupe_interval,
                                 f"No complete traces available for window {window_start} (attempt {window_retry_count[window_key]}/{max_retries})")
                if buffer_traces > 0:
                    logger.warning(f"Buffer has {buffer_traces} traces but none cover the required window")
                # Don't advance window_start - retry this window in the next iteration
                # Increase sleep to back off on repeated failures
                sleep_time = min(5 * window_retry_count[window_key], 60)
                time.sleep(sleep_time)
                continue
            
            # Create output directory
            dirname = window_start.strftime('%Y%m%d%H%M%S')
            outdir = os.path.join(project_folder, dirname)
            os.makedirs(outdir, exist_ok=True)
            
            logger.info(f"Saving {len(fixed_traces)} fixed-length traces for window {window_start}")
            
            # Save all traces with identical time windows
            for tr in fixed_traces:
                # Quality control - check for spikes
                try:
                    if spike_detection(tr, spike_percent, spike_multiplier, spike_threshold):
                        logger.warning(f"Trace {tr.id} rejected due to spikes")
                        continue
                except Exception as e:
                    logger.warning(f"Spike detection failed for {tr.id}: {e}")
                
                # Create filename with exact window timing
                window_end = window_start + window_length
                starttime_str = window_start.strftime('%Y%m%d%H%M%S')
                endtime_str = window_end.strftime('%Y%m%d%H%M%S')
                
                # Use actual location code or empty string if not specified
                location = tr.stats.location if tr.stats.location else ""
                if location:
                    filename = f"{tr.stats.network}.{tr.stats.station}.{location}.{tr.stats.channel}__{starttime_str}__{endtime_str}.mseed"
                else:
                    filename = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}__{starttime_str}__{endtime_str}.mseed"
                filepath = os.path.join(outdir, filename)
                
                try:
                    tr.write(filepath, format='MSEED')
                    logger.debug(f"Saved: {filename} ({len(tr.data)} samples)")
                except Exception as e:
                    logger.error(f"Failed to save {filename}: {e}")
            
            # Copy inventory file with error handling (like in v3)
            maininv = os.path.join(project_folder, 'rt.xml')
            try:
                if os.path.exists(maininv):
                    import shutil
                    shutil.copy2(maininv, outdir)
                    logger.debug(f"Copied rt.xml to {outdir}")
                else:
                    logger.warning(f"Main inventory file {maininv} not found, skipping copy")
            except Exception as e:
                logger.warning(f"Could not copy inventory file: {e}")
            
            # Clean up old data from buffer
            cleanup_buffer(data_buffer, window_start, buffer_length, lock)
            
            # Log buffer status
            with lock:
                buffer_status = {trace_id: len(stream) for trace_id, stream in data_buffer.items() if len(stream) > 0}
                logger.info(f"Buffer contains {len(buffer_status)} traces, total packets: {client.packet_count}")
            
            last_window_start = window_start
            
    except KeyboardInterrupt:
        logger.info("Stopping SeedLink download...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise

def sds_fixed_window_download(
    sds_root,
    stalist,
    project_folder='/scratch/realtime',
    window_length=180,
    window_overlap=30,
    starttime=None,
    endtime=None,
    spike_percent=0.95,
    spike_multiplier=2,
    spike_threshold=5
):
    """
    Extract fixed-length windows from an SDS-style filesystem archive using ObsPy's SDS client.

    Parameters:
    -----------
    sds_root : str
        Path to the SDS archive root (top-level directory containing NET/STA/...) or a URL supported by SDSClient.
    stalist : list
        List of station selectors in format 'NET_STA:CHAN' (same format used elsewhere in this module).
    project_folder : str
        Directory to save waveform data (default: '/scratch/realtime')
    window_length : int
        Window length in seconds (default: 180 = 3 minutes)
    window_overlap : int
        Overlap in seconds between consecutive windows (default: 30 seconds)
        Each window starts on the minute, but includes overlap with previous window
    starttime : UTCDateTime or None
        Start time for processing (defaults to now-3600s)
    endtime : UTCDateTime or None
        End time for processing (defaults to now)
    Returns:
    --------
    None
    """
    # Create client
    try:
        client = SDSClient(sds_root)
    except Exception as e:
        logger.error(f"Could not initialize SDS client for {sds_root}: {e}")
        return

    if starttime is None:
        starttime = UTCDateTime.now() - 3600
    if endtime is None:
        endtime = UTCDateTime.now()

    # Align starttime to minute boundary
    minute_boundary = 60
    aligned_start = UTCDateTime(int((starttime.timestamp) // minute_boundary) * minute_boundary)
    window_start = aligned_start
    
    # Calculate the step between windows (window length minus overlap)
    window_step = window_length - window_overlap

    logger.info(f"Starting SDS fixed-window extraction from {sds_root} between {starttime} and {endtime}")
    logger.info(f"Window configuration: {window_length}s windows with {window_overlap}s overlap, stepping every {window_step}s")

    while window_start + window_length <= endtime:
        window_end = window_start + window_length
        logger.info(f"Processing window {window_start} to {window_end}")

        # Create output directory
        dirname = window_start.strftime('%Y%m%d%H%M%S')
        outdir = os.path.join(project_folder, dirname)
        os.makedirs(outdir, exist_ok=True)

        saved = 0
        for sel in stalist:
            try:
                # Expected format NET_STA:CHAN (network_station:channel)
                net = sel.split('_')[0]
                rest = sel.split('_', 1)[1]
                station = rest.split(':')[0]
                channel = rest.split(':')[1] if ':' in rest else '*'
                location = ''  # let SDS client return whatever location exists

                # Try to fetch waveforms for this selector
                try:
                    st = client.get_waveforms(net, station, location, channel, window_start, window_end)
                except Exception:
                    # Some SDS layouts don't like empty location; try None
                    try:
                        st = client.get_waveforms(net, station, None, channel, window_start, window_end)
                    except Exception as e:
                        logger.debug(f"No data for {sel} in window {window_start}: {e}")
                        continue

                if st is None or len(st) == 0:
                    logger.debug(f"Empty stream for {sel} in window {window_start}")
                    continue

                # Normalize and trim each trace to exact window
                for tr in st:
                    try:
                        fixed_tr = tr.copy()
                        fixed_tr.trim(starttime=window_start, endtime=window_end, pad=True, fill_value=0)

                        # Ensure exact npts
                        expected_samples = int(window_length * tr.stats.sampling_rate)
                        if len(fixed_tr.data) != expected_samples:
                            if abs(len(fixed_tr.data) - expected_samples) <= 2:
                                if len(fixed_tr.data) < expected_samples:
                                    padding = expected_samples - len(fixed_tr.data)
                                    fixed_tr.data = np.concatenate([fixed_tr.data, np.zeros(padding)])
                                else:
                                    fixed_tr.data = fixed_tr.data[:expected_samples]
                            else:
                                logger.warning(f"Sample count mismatch for {tr.id} from SDS: got {len(fixed_tr.data)}, expected {expected_samples}")
                                continue

                        fixed_tr.stats.starttime = window_start
                        fixed_tr.stats.npts = len(fixed_tr.data)

                        # Quality control
                        if spike_detection(fixed_tr, spike_percent, spike_multiplier, spike_threshold):
                            logger.warning(f"Trace {fixed_tr.id} rejected due to spikes")
                            continue

                        # Build filename
                        starttime_str = window_start.strftime('%Y%m%d%H%M%S')
                        endtime_str = window_end.strftime('%Y%m%d%H%M%S')
                        location_code = fixed_tr.stats.location if fixed_tr.stats.location else ''
                        if location_code:
                            filename = f"{fixed_tr.stats.network}.{fixed_tr.stats.station}.{location_code}.{fixed_tr.stats.channel}__{starttime_str}__{endtime_str}.mseed"
                        else:
                            filename = f"{fixed_tr.stats.network}.{fixed_tr.stats.station}.{fixed_tr.stats.channel}__{starttime_str}__{endtime_str}.mseed"

                        filepath = os.path.join(outdir, filename)
                        fixed_tr.write(filepath, format='MSEED')
                        saved += 1
                        logger.debug(f"Saved {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to process trace {tr.id} from SDS: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Error handling selector {sel}: {e}")
                continue

        if saved == 0:
            logger.info(f"No traces saved for window {window_start}")
        else:
            logger.info(f"Saved {saved} traces for window {window_start}")

        # Advance to next window
        window_start = window_start + window_length

    logger.info("SDS fixed-window extraction completed")

def expand_sds_selectors(sds_root, selector_pattern):
    """
    Expand a selector pattern (may contain shell-style wildcards) into a list
    of explicit selectors in the form 'NET_STA:CHAN' by scanning the SDS root.

    This assumes a simple SDS layout of sds_root/NET/STA/CHAN/..., which is
    common. If your SDS/SeisComP layout differs (e.g., an extra location
    subdirectory), this function may need to be adapted.
    """
    selectors = set()
    try:
        if not os.path.isdir(sds_root):
            logger.warning(f"SDS root {sds_root} is not a directory")
            return []

        for net in os.listdir(sds_root):
            net_path = os.path.join(sds_root, net)
            if not os.path.isdir(net_path):
                continue
            for sta in os.listdir(net_path):
                sta_path = os.path.join(net_path, sta)
                if not os.path.isdir(sta_path):
                    continue
                # Try to find channel directories under station
                for maybe_chan in os.listdir(sta_path):
                    chan_path = os.path.join(sta_path, maybe_chan)
                    if not os.path.isdir(chan_path):
                        continue
                    candidate = f"{net}_{sta}:{maybe_chan}"
                    if fnmatch.fnmatch(candidate, selector_pattern):
                        selectors.add(candidate)
                # Also check one more level in case layout is NET/STA/LOC/CHAN
                for loc in os.listdir(sta_path):
                    loc_path = os.path.join(sta_path, loc)
                    if not os.path.isdir(loc_path):
                        continue
                    for maybe_chan in os.listdir(loc_path):
                        chan_path = os.path.join(loc_path, maybe_chan)
                        if not os.path.isdir(chan_path):
                            continue
                        candidate = f"{net}_{sta}:{maybe_chan}"
                        if fnmatch.fnmatch(candidate, selector_pattern):
                            selectors.add(candidate)
    except Exception as e:
        logger.warning(f"Error expanding SDS selectors: {e}")

    return sorted(selectors)


def load_config(config_path):
    """
    Load a YAML configuration file describing downloads.

    Expected format:
      downloads:
        - type: sds
          sds_root: /path/to/sds
          selectors: ["NET_STA:HHZ", "NET_*:HHZ"]
          project_folder: /scratch/realtime
          window_length: 300
          starttime: '2025-09-03T00:00:00'
          endtime: '2025-09-03T01:00:00'
        - type: seedlink
          servers: ['rtserve.ou.edu:18000']
          selectors: ['*']
          project_folder: /scratch/realtime
          window_length: 60

    Returns dict of parsed config or None on error.
    """
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            logger.error(f"Config file {config_path} parsed to non-dict")
            return None
        return cfg
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        return None
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return None

def spike_detection(trace, spike_percent=0.95, spike_multiplier=2, spike_threshold=5):
    """
    Detect spikes in seismic data.
    
    Parameters:
    -----------
    trace : obspy.Trace
        Input trace
    spike_percent : float
        Percentile for baseline calculation
    spike_multiplier : float
        Multiplier for spike threshold
    spike_threshold : int
        Maximum number of spikes allowed
        
    Returns:
    --------
    bool : True if trace should be rejected due to spikes
    """
    try:
        data = trace.data
        if len(data) == 0:
            return True
            
        # Calculate baseline using percentile
        baseline = np.percentile(np.abs(data), spike_percent * 100)
        threshold = baseline * spike_multiplier
        
        # Count spikes
        spikes = np.sum(np.abs(data) > threshold)
        
        return spikes > spike_threshold
    except Exception:
        return False

def discover_stations(server, timeout=10):
    """
    Discover available stations on a SeedLink server by parsing INFO request response.
    
    Parameters:
    -----------
    server : str
        SeedLink server address (e.g., 'rtserve.ou.edu:18000')
    timeout : int
        Timeout in seconds for the discovery process
        
    Returns:
    --------
    list : List of station codes in format 'NET_STA:CHAN'
    """
    try:
        # Create a temporary buffer to capture discovery info
        discovery_buffer = defaultdict(Stream)
        discovery_lock = threading.Lock()

        # Create a SeedLink client for discovery
        client = WaveformBuffer(discovery_buffer, lock=discovery_lock)
        client.slconn.set_sl_address(server)
        client.multiselect = '*'
        client.verbose = 0  # Reduce verbosity for discovery

        stations = []
        try:
            client.initialize()

            # Start client briefly to discover stations
            start_time = time.time()
            discovery_thread = threading.Thread(target=client.run, daemon=True)
            discovery_thread.start()

            # Wait for some data to arrive or timeout
            time.sleep(min(timeout, 5))

            # Extract station information from buffer
            with discovery_lock:
                for trace_id in discovery_buffer.keys():
                    try:
                        # Parse trace ID (should be NET.STA.LOC.CHAN)
                        parts = trace_id.split('.')
                        if len(parts) >= 4:
                            net, sta, loc, chan = parts[0], parts[1], parts[2], parts[3]
                            station_code = f"{net}_{sta}:{chan}"
                            if station_code not in stations:
                                stations.append(station_code)
                    except Exception as e:
                        logger.debug(f"Error parsing trace ID {trace_id}: {e}")
                        continue

            logger.info(f"Discovered {len(stations)} stations on {server}")
            return sorted(stations)

        except Exception as e:
            logger.error(f"Error during station discovery on {server}: {e}")
            return []
            
    except Exception as e:
        logger.error(f"Failed to initialize discovery client for {server}: {e}")
        return []

def start_multiple_seedlink_downloads(
    servers,
    stalist,
    project_folder='/home/jwalter/scratch',
    window_length=300
):
    """
    Start continuous_fixed_window_download for multiple SeedLink servers in separate threads.
    servers: list of server addresses (e.g., ['rtserve.ou.edu:18000', 'other.server:18000'])
    stalist: list of station selectors (same list used for all servers)
    """
    threads = []
    for server in servers:
        t = threading.Thread(
            target=continuous_fixed_window_download,
            kwargs={
                'server': server,
                'stalist': stalist,
                'project_folder': project_folder,
                'window_length': window_length
            },
            daemon=True
        )
        t.start()
        logger.info(f"Started SeedLink download thread for {server}")
        threads.append(t)
    return threads

def cleanup_old_folders(project_folder, max_age_hours=48, check_interval_hours=1):
    """
    Periodically check and delete old folders that don't have associated events.
    
    Parameters:
    -----------
    project_folder : str
        Base directory containing timestamp folders
    max_age_hours : int
        Maximum age in hours before a folder is considered for deletion (default: 48)
    check_interval_hours : int
        How often to run the cleanup check in hours (default: 1)
    """
    def cleanup_worker():
        while True:
            try:
                logger.info("Starting cleanup check for old folders...")
                now = UTCDateTime()
                cutoff_time = now - (max_age_hours * 3600)  # Convert hours to seconds
                
                # Get all timestamp directories with better error handling
                try:
                    if not os.path.exists(project_folder):
                        logger.error(f"Project folder {project_folder} does not exist!")
                        time.sleep(check_interval_hours * 3600)
                        continue
                        
                    all_dirs = [d for d in os.listdir(project_folder) 
                              if os.path.isdir(os.path.join(project_folder, d)) 
                              and d.isdigit() and len(d) == 14]
                    logger.debug(f"Found {len(all_dirs)} timestamp directories")
                except Exception as e:
                    logger.warning(f"Could not list project folder for cleanup: {e}")
                    time.sleep(check_interval_hours * 3600)
                    continue
                
                # Get all event XML files to check which folders have events
                try:
                    event_xml_files = glob.glob(os.path.join(project_folder, "*_seiscomp.xml"))
                except Exception as e:
                    logger.warning(f"Could not search for event XML files: {e}")
                    event_xml_files = []
                    
                folders_with_events = set()
                
                for xml_file in event_xml_files:
                    try:
                        if not os.path.exists(xml_file):
                            continue  # File may have been deleted during processing
                            
                        filename = os.path.basename(xml_file)
                        # Extract timestamp from filename (assuming format contains timestamp)
                        # Look for 14-digit timestamp patterns in the filename
                        import re
                        timestamp_matches = re.findall(r'\d{14}', filename)
                        for timestamp in timestamp_matches:
                            if timestamp in all_dirs:
                                folders_with_events.add(timestamp)
                                logger.debug(f"Found event for folder {timestamp}: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not process event file {xml_file}: {e}")
                        continue
                
                logger.info(f"Found {len(folders_with_events)} folders with associated events")
                
                deleted_count = 0
                preserved_count = 0
                
                for dirname in all_dirs:
                    try:
                        # Parse timestamp from directory name
                        dir_time = UTCDateTime(f"{dirname[0:4]}-{dirname[4:6]}-{dirname[6:8]}T{dirname[8:10]}:{dirname[10:12]}:{dirname[12:14]}")
                        
                        # Check if folder is old enough and has no associated events
                        if dir_time < cutoff_time and dirname not in folders_with_events:
                            folder_path = os.path.join(project_folder, dirname)
                            
                            # Additional safety check: make sure folder actually exists
                            if os.path.exists(folder_path):
                                # Get folder size for logging
                                try:
                                    folder_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                                    for dirpath, dirnames, filenames in os.walk(folder_path)
                                                    for filename in filenames)
                                    folder_size_mb = folder_size / (1024 * 1024)
                                except:
                                    folder_size_mb = 0
                                
                                logger.info(f"Deleting old folder: {dirname} (age: {(now - dir_time)/3600:.1f}h, size: {folder_size_mb:.1f}MB)")
                                
                                # Use try-except for the actual deletion
                                try:
                                    shutil.rmtree(folder_path)
                                    deleted_count += 1
                                except Exception as e:
                                    logger.warning(f"Could not delete folder {folder_path}: {e}")
                            else:
                                logger.debug(f"Folder {dirname} already deleted")
                        else:
                            preserved_count += 1
                            if dirname in folders_with_events:
                                logger.debug(f"Preserving folder {dirname} - has associated event")
                            elif dir_time >= cutoff_time:
                                logger.debug(f"Preserving folder {dirname} - too recent (age: {(now - dir_time)/3600:.1f}h)")
                                
                    except Exception as e:
                        logger.warning(f"Could not process folder {dirname} for cleanup: {e}")
                        continue
                
                logger.info(f"Cleanup completed: deleted {deleted_count} folders, preserved {preserved_count} folders")
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
            
            # Sleep until next cleanup check
            time.sleep(check_interval_hours * 3600)
    
    # Start cleanup worker in a daemon thread
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info(f"Started cleanup worker - will check every {check_interval_hours}h and delete folders older than {max_age_hours}h without events")
    return cleanup_thread

# Example usage in __main__: adds CLI with --config support
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SeedLink / SDS fixed-window downloader')
    parser.add_argument('--config', '-c', help='YAML config file describing downloads', default=None)
    parser.add_argument('--seedlink', action='store_true', help='Run default SeedLink downloader')
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        if not cfg:
            logger.error('Failed to load config, exiting')
            raise SystemExit(1)

        downloads = cfg.get('downloads', [])
        if len(downloads) == 0:
            logger.error('No downloads found in config')
            raise SystemExit(1)

        # Execute each job in the config
        for job in downloads:
            jtype = job.get('type')
            if jtype == 'sds':
                sds_root = job.get('sds_root')
                selectors = job.get('selectors', [])
                expanded = []
                for sel in selectors:
                    if any(ch in sel for ch in ['*', '?', '[']):
                        expanded += expand_sds_selectors(sds_root, sel)
                    else:
                        expanded.append(sel)

                start = None
                end = None
                if job.get('starttime'):
                    start = UTCDateTime(job.get('starttime'))
                if job.get('endtime'):
                    end = UTCDateTime(job.get('endtime'))

                # Run synchronously (blocking) by default
                sds_fixed_window_download(
                    sds_root=sds_root,
                    stalist=expanded,
                    project_folder=job.get('project_folder', '/scratch/realtime'),
                    window_length=job.get('window_length', 300),
                    starttime=start,
                    endtime=end
                )

            elif jtype == 'seedlink':
                servers = job.get('servers', [])
                selectors = job.get('selectors', ['*'])
                # Start seedlink downloads
                start_multiple_seedlink_downloads(
                    servers=servers,
                    stalist=selectors,
                    project_folder=job.get('project_folder', '/scratch/realtime'),
                    window_length=job.get('window_length', 60)
                )

                # Start cleanup worker if requested
                cleanup_old_folders(
                    project_folder=job.get('project_folder', '/scratch/realtime'),
                    max_age_hours=job.get('max_age_hours', 48),
                    check_interval_hours=job.get('check_interval_hours', 1)
                )

            else:
                logger.warning(f"Unknown job type in config: {jtype}")

        # If any seedlink jobs were started, keep main thread alive
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info('Shutting down per user interrupt')

    elif args.seedlink:
    # Run original SeedLink discovery and start downloads
        now = UTCDateTime()
        try:
            client = EasySeedLinkClient('rtserve.ou.edu')
            streams_xml = client.get_info('STREAMS')
            root = ET.fromstring(streams_xml)
            stalist = []
            print('Available stations from SeedLink server:')
            for child in root:
                net = child.attrib['network']
                sta = child.attrib['name']
                print(f'  Network: {net}, Station: {sta}')
                for streams1 in child.findall('.//stream'):
                    if UTCDateTime(streams1.attrib['end_time']) > (now - 3600):
                        if streams1.attrib['seedname'][-1] in {'E', 'N', '1', '2', 'Z'}:
                            if len(child.findall('.//stream')) > 4:
                                if streams1.attrib['seedname'][0] in {'H', 'B'} and streams1.attrib['seedname'][-2] == 'H':
                                    station_channel = net + '_' + sta + ':' + streams1.attrib['seedname']
                                    stalist.append(station_channel)
                                    print(f'    Added: {station_channel}')
                            else:
                                station_channel = net + '_' + sta + ':' + streams1.attrib['seedname']
                                stalist.append(station_channel)
                                print(f'    Added: {station_channel}')
            print(f"\nTotal stations/channels selected: {len(stalist)}")
        except Exception as e:
            print(f'Could not get station list from server: {e}')
            stalist = ['OK_CROK:HHZ', 'OK_CROK:HHE', 'OK_CROK:HHN']
            print(f'Using fallback station list: {stalist}')

        project_folder = '/scratch/realtime'
        start_multiple_seedlink_downloads(
            servers=['rtserve.ou.edu:18000'],
            stalist=stalist,
            project_folder=project_folder,
            window_length=60
        )
        cleanup_old_folders(project_folder=project_folder, max_age_hours=48, check_interval_hours=1)

        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info('Shutting down all SeedLink downloader threads.')

    else:
        parser.print_help()