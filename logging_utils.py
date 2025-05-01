import logging
import json
import uuid
import os
import datetime
import traceback
from logging.handlers import RotatingFileHandler # Import RotatingFileHandler
import config # Import config to access settings

# Global variable to hold the logger instance once configured
_simulation_logger = None

class JsonFormatter(logging.Formatter):
    """
    Formats log records as JSON Lines.
    """
    def format(self, record):
        log_entry = {
            'timestamp_iso': datetime.datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'logger_name': record.name,
        }
        # If the message is already a dict (from log_event), merge it
        if isinstance(record.msg, dict):
            # Ensure we don't overwrite core fields unless intended
            original_message = log_entry.get('message') # Store original message
            log_entry.update(record.msg)
            # Restore essential fields if overwritten by the dict
            log_entry['level'] = record.levelname
            log_entry['logger_name'] = record.name
            # Keep original formatted message if the dict didn't have 'message'
            if 'message' not in record.msg and original_message:
                 log_entry['message'] = original_message

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = traceback.format_exception(*record.exc_info)
            # Add exception message to main message field for easier viewing
            if 'message' not in log_entry or not log_entry['message']:
                 log_entry['message'] = str(record.exc_info[1])


        return json.dumps(log_entry, default=str) # Use default=str for non-serializable types

def setup_logger(log_dir, log_level=None, log_filename=None, max_bytes=None, backup_count=None):
    """
    Configures the root logger to output JSON Lines to a rotating file
    and standard format to the console (for INFO level and above).
    Uses settings from config.py by default.

    Args:
        log_dir (str): The directory to save the log file in.
        log_level (int, optional): The minimum logging level. Defaults to config.LOG_LEVEL.
        log_filename (str, optional): The name for the JSON Lines log file. Defaults to config.LOG_FILENAME.
        max_bytes (int, optional): Max bytes per log file for rotation. Defaults to config.LOG_MAX_BYTES.
        backup_count (int, optional): Number of backup files. Defaults to config.LOG_BACKUP_COUNT.


    Returns:
        logging.Logger: The configured logger instance.
    """
    global _simulation_logger
    if _simulation_logger:
        return _simulation_logger

    # Use config values if arguments are not provided
    log_level = log_level if log_level is not None else config.LOG_LEVEL
    log_filename = log_filename if log_filename is not None else config.LOG_FILENAME
    max_bytes = max_bytes if max_bytes is not None else config.LOG_MAX_BYTES
    backup_count = backup_count if backup_count is not None else config.LOG_BACKUP_COUNT

    logger = logging.getLogger('simulation') # Use a specific name
    logger.setLevel(log_level)

    # Prevent propagation to default root logger
    logger.propagate = False

    # Remove existing handlers if any (useful for re-runs in notebooks)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # --- Rotating JSON File Handler ---
    log_filepath = os.path.join(log_dir, log_filename)
    # Use RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_filepath,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level) # Log everything at the specified level to the file
    json_formatter = JsonFormatter()
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    # --- Console Handler (optional, for higher level messages) ---
    console_handler = logging.StreamHandler()
    # Set console handler to WARNING level to reduce verbosity
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    _simulation_logger = logger
    logger.info(f"Logger initialized. Saving JSON logs to: {log_filepath} with rotation.")
    return logger

def log_event(step, event_type, agent_id, details=None):
    """
    Logs a structured event to the simulation logger.

    Args:
        step (int): The current simulation step number.
        event_type (str): A string identifying the type of event (e.g., 'artifact_generated').
        agent_id (int or str): The ID of the agent primarily involved.
        details (dict, optional): A dictionary containing event-specific information.
    """
    global _simulation_logger
    if not _simulation_logger:
        print("ERROR: Logger not initialized. Call setup_logger first.")
        return

    log_payload = {
        'simulation_step': step,
        'event_id': uuid.uuid4().hex,
        'event_type': event_type,
        'agent_id': agent_id,
        'details': details or {}
    }

    # Use the logger's info method; the formatter will handle JSON conversion
    _simulation_logger.info(log_payload)

# Example usage (won't run directly here, but shows intent)
if __name__ == '__main__':
    # In a real script, you'd call setup_logger first
    # logger = setup_logger('temp_logs')
    # log_event(step=1, event_type='test_event', agent_id=0, details={'value': 42})
    pass
