#!/usr/bin/env python3
"""
SeedLink Connection Script v4 - Fixed Window Processing
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

Author: Modified from original SeedLink example
Date: 2024
"""

import os
import time
import threading
import numpy as np
from obspy import Stream, UTCDateTime, read, Trace
from obspy.clients.seedlink import SLClient
import logging
import glob
import shutil
from collections import defaultdict
import xml.etree.ElementTree as ET
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Silence extremely verbose/info messages from obspy's seedlink connection
# (they include a formatting bug and duplicate 0x prefixes causing noisy logs)
logging.getLogger('obspy.clients.seedlink.client.seedlinkconnection').setLevel(logging.WARNING)

# Simple helper for rate-limited logging to avoid repeating identical messages
_last_log_time = {}
def rate_limited_log(level, key, interval, msg, *args, **kwargs):
    """Log a message at `level` at most once per `interval` seconds for `key`.

    key should be stable for messages that must be deduplicated (for example
    a particular window start/time). A small helper to reduce log spam.
    """
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

# Alerting helpers
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
    """Send a simple Slack incoming-webhook alert.

    Slack incoming webhooks expect JSON with a 'text' field (or blocks). We keep
    it simple and send a formatted text payload. This function is separated so
    we can keep Slack-specific formatting without changing the generic webhook
    helper.
    """
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
    """Send an alert, deduped by key for interval seconds.

    Behavior:
    - If alert_callback is provided, call it with (subject, msg_text)
    - Else if alert_webhook is provided, POST JSON {subject, message}
    - Else if alert_email is provided (dict), attempt SMTP send
    - Else, fallback to logger.warning
    """
    # rate-limit alerts
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

    # Slack-specific webhook has a different payload format (text) so handle it first
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

    # fallback to logger
    logger.warning("ALERT: %s - %s", subject, msg_text)
    return True

class SeedlinkBuffer(SLClient):
    """
    SeedLink client that buffers all incoming data for fixed-window processing
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
            # Be more lenient - allow traces that mostly cover the window
            has_overlap = not (trace_end <= window_start or trace_start >= window_end)
            covers_start = trace_start <= window_start + 10  # Allow 10 second tolerance
            covers_end = trace_end >= window_end - 10  # Allow 10 second tolerance
            
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
                logger.debug(f"  has_overlap={has_overlap}, covers_start={covers_start} (need <={window_start + 10}), covers_end={covers_end} (need >={window_end - 10})")
    
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
    # Add extra margin to avoid cleaning data we might still need
    safety_margin = 300  # Keep 5 extra minutes
    cutoff_time = window_start - max_buffer_length - safety_margin
    
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
    # Watchdog / diagnostics options
    stall_threshold=180,
    dedupe_interval=60,
    restart_backoff_initial=5,
    restart_backoff_max=300,
    # Optional alerting: callback(webhook/email/fallback)
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
        Seconds to suppress duplicate identical log messages (default: 60)
    restart_backoff_initial : int
        Initial backoff seconds for restarting the SeedLink client (default: 5)
    restart_backoff_max : int
        Maximum backoff seconds between restart attempts (default: 300)
    """
    lock = threading.Lock()
    data_buffer = defaultdict(Stream)  # Dictionary to store traces by ID
    
    client = SeedlinkBuffer(data_buffer, lock=lock)
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
    # reduce obspy SeedLink client verbosity to avoid noisy INFO messages
    # (the upstream seedlinkconnection module prints malformed INFO lines)
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
    
    # Wait for initial data to accumulate
    # Need to wait at least 2x window_length to have complete windows available
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
        # Notify operator if requested
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
    # Go back enough to ensure data availability (at least 3x window_length)
    aligned_start = UTCDateTime(int((current_time.timestamp - 3 * window_length) // minute_boundary) * minute_boundary)
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
            # Need to wait for the full window + additional buffer to ensure all data arrived
            min_window_age = max(90, window_length / 2)  # At least 90s or half the window length
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
                # Alert operator once for thread deaths (rate-limited)
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
                    # Reset the backoff on progress
                    current_backoff = restart_backoff_initial
                elif time.time() - last_progress_time > stall_threshold:
                    # Only perform a restart if backoff window has passed
                    if time.time() - last_restart_time >= current_backoff:
                        logger.warning("No new packets for %ds; restarting SeedLink client (backoff %ds)", stall_threshold, current_backoff)
                        # Alert operator when watchdog restarts the client (rate-limited)
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
                    # Alert operator about giving up on a window
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
                # But add sleep to avoid tight loop; increase sleep proportional to attempts
                sleep_time = min(5 * window_retry_count[window_key], 60)
                time.sleep(sleep_time)
                continue
            
            # Create output directory
            dirname = window_start.strftime('%Y%m%d%H%M%S')
            outdir = os.path.join(project_folder, dirname)
            os.makedirs(outdir, exist_ok=True)
            
            # Clean up retry counter for this successful window
            window_key = str(window_start)
            if window_key in window_retry_count:
                del window_retry_count[window_key]
            
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
        client = SeedlinkBuffer(discovery_buffer, lock=discovery_lock)
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
    window_length=180,
    window_overlap=30
):
    """
    Start continuous_fixed_window_download for multiple SeedLink servers in separate threads.
    servers: list of server addresses (e.g., ['rtserve.ou.edu:18000', 'other.server:18000'])
    stalist: list of station selectors (same list used for all servers)
    window_length: window length in seconds (default: 180 = 3 minutes)
    window_overlap: overlap in seconds between consecutive windows (default: 30 seconds)
    """
    threads = []
    for server in servers:
        t = threading.Thread(
            target=continuous_fixed_window_download,
            kwargs={
                'server': server,
                'stalist': stalist,
                'project_folder': project_folder,
                'window_length': window_length,
                'window_overlap': window_overlap
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

# Example usage in __main__:
if __name__ == "__main__":
    import xml.etree.ElementTree as ET
    from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
    from obspy.clients.fdsn import Client
    from obspy import Inventory
    
    now = UTCDateTime()
    
    # Get all available stations from SeedLink server
    try:
        client = EasySeedLinkClient('rtserve.ou.edu')
        streams_xml = client.get_info('STREAMS')
        root = ET.fromstring(streams_xml)
        stalist = []
        
        print("Available stations from SeedLink server:")
        for child in root:
            net = child.attrib['network']
            sta = child.attrib['name']
            print(f"  Network: {net}, Station: {sta}")
            
            for streams1 in child.findall('.//stream'):
                if UTCDateTime(streams1.attrib['end_time']) > (now - 3600):
                    if streams1.attrib['seedname'][-1] in {'E', 'N', '1', '2', 'Z'}:
                        if len(child.findall('.//stream')) > 4:
                            if streams1.attrib['seedname'][0] in {'H', 'B'} and streams1.attrib['seedname'][-2] == 'H':
                                station_channel = net + '_' + sta + ':' + streams1.attrib['seedname']
                                stalist.append(station_channel)
                                print(f"    Added: {station_channel}")
                        else:
                            station_channel = net + '_' + sta + ':' + streams1.attrib['seedname']
                            stalist.append(station_channel)
                            print(f"    Added: {station_channel}")
        
        print(f"\nTotal stations/channels selected: {len(stalist)}")
        
    except Exception as e:
        print(f"Could not get station list from server: {e}")
        # Fallback to a basic list
        stalist = ['OK_CROK:HHZ', 'OK_CROK:HHE', 'OK_CROK:HHN']
        print(f"Using fallback station list: {stalist}")

    project_folder = '/scratch/realtime'
    
    # Create inventory file if it doesn't exist
    if not os.path.exists(project_folder+'/rt.xml'):
        try:
            fdsnclient = Client()
            inv = Inventory()
            for sta in stalist:
                try:
                    network = sta.split('_')[0]
                    station = sta.split('_')[1].split(':')[0]
                    inv += fdsnclient.get_stations(
                        starttime=now-5,
                        endtime=now,
                        network=network,
                        station=station,
                        channel='*H*',
                        level='response'
                    )
                except Exception as e:
                    print(f"Could not get station metadata for {sta}: {e}")
            
            if len(inv) > 0:
                inv.write(project_folder+'/rt.xml','STATIONXML')
                print(f"Created inventory file with {len(inv)} stations")
            else:
                print("No station metadata available")
        except Exception as e:
            print(f"Could not create inventory file: {e}")

    # List of SeedLink servers to connect to
    servers = [
        'rtserve.ou.edu:18000',
        # Add more servers as needed, e.g.:
        # 'other.seedlink.server:18000',
    ]

    print(f"\nStarting SeedLink downloads for {len(stalist)} stations...")
    print(f"Project folder: {project_folder}")
    print(f"Window configuration: 3 minute (180s) windows with 30s overlap")
    
    # Start downloads for all servers in parallel
    start_multiple_seedlink_downloads(
        servers=servers,
        stalist=stalist,
        project_folder=project_folder,
        window_length=180,  # 3 minutes
        window_overlap=30    # 30 seconds overlap
    )

    # Start cleanup worker to delete old folders without events
    cleanup_old_folders(
        project_folder=project_folder,
        max_age_hours=48,  # Delete folders older than 48 hours
        check_interval_hours=1  # Check every hour
    )

    # Keep main thread alive
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down all SeedLink downloader threads.")