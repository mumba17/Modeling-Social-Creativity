# Logging Overview (Updated April 24, 2025)

This document outlines the centralized logging mechanisms used throughout the `Modeling-Social-Creativity` codebase. The system is designed to provide comprehensive, structured logs for analysis and debugging, primarily using Python's built-in `logging` module configured to output JSON Lines (JSONL).

## Centralized Logging System (`logging_utils.py`)

The core of the logging system resides in `logging_utils.py`, which provides:

1.  **`setup_logger(log_dir, ...)`:**
    *   Configures a main logger named 'simulation'.
    *   Sets up a `logging.handlers.RotatingFileHandler` to write logs to a JSON Lines file (e.g., `simulation_events.jsonl`) within the specified `log_dir`.
    *   Implements log rotation based on file size (`LOG_MAX_BYTES`) and backup count (`LOG_BACKUP_COUNT`) defined in `config.py`.
    *   Uses a custom `JsonFormatter` to ensure all file log entries are valid JSON objects, one per line.
    *   Optionally sets up a `logging.StreamHandler` to output INFO level (or higher) messages to the console in a standard format for real-time feedback.
    *   Reads default configuration (log level, filename, rotation settings) from `config.py`.

2.  **`log_event(step, event_type, agent_id, details)`:**
    *   A helper function to log structured events.
    *   Takes simulation step, a specific `event_type` string, optional `agent_id`, and a `details` dictionary.
    *   Logs this information as a dictionary message using the 'simulation' logger at the INFO level. The `JsonFormatter` automatically incorporates these fields into the final JSON log entry.

## Logging Practices by Module

*   **`model.py`:**
    *   Initializes the centralized logger using `setup_logger()`.
    *   Uses `log_event()` extensively to record key simulation events:
        *   `agent_step_summary`: Agent's state at the start of its step.
        *   `domain_interaction_evaluation`: When an agent evaluates an artifact from the domain.
        *   `domain_adoption`: When an agent adopts an artifact from the domain.
        *   `message_evaluation`: When an agent evaluates a received message.
        *   `artifact_generated`: When an agent generates a new artifact (mutation or breeding).
        *   `message_sent`: When an agent shares an artifact.
    *   Uses standard `logger.warning`, `logger.error` for issues during simulation steps (e.g., processing errors).
    *   Manages logger cleanup (`handler.close()`) at the end of the simulation run.
    *   Still uses a `SummaryWriter` for TensorBoard logging of high-level scalar metrics (e.g., average interest, thresholds, network stats).

*   **`genart.py`:**
    *   Uses `log_event()` within `ExpressionNode.mutate` and `ExpressionNode.breed` (called with context from `model.py`) to log:
        *   `mutation_applied`
        *   `breed_operation`
    *   Uses standard `logger.warning`, `logger.error` for issues during expression evaluation or image generation.

*   **`knn.py`:**
    *   Uses `log_event()` within `kNN.add_feature_vectors` (called with context from `model.py`) to log:
        *   `knn_vectors_added`
        *   `knn_add_nan_skipped`
        *   `knn_add_error`
    *   Uses standard `logger.debug`, `logger.warning`, `logger.error` for internal kNN operations (e.g., elbow calculation, index building, cache status).

*   **`timing_utils.py`:**
    *   Uses the standard `logging.getLogger('simulation')` (obtained via `logging_utils`) within `TimingStats.print_step_report` to log timing summaries at the INFO level.

*   **Other Modules (`features.py`, `network_tracker.py`, etc.):**
    *   Use standard `logging.getLogger(__name__)` for module-specific informational messages, warnings, or errors. These messages are captured by the handlers configured in `setup_logger`.

## Log Output

*   **JSON Lines File (e.g., `logs/run_YYYYMMDD_HHMMSS/simulation_events.jsonl`):** Contains detailed, structured event data logged via `log_event` and standard log messages formatted as JSON. Suitable for programmatic analysis (e.g., using Pandas `read_json(lines=True)`). Rotates automatically.
*   **Console Output:** Displays INFO level (or higher) messages from the 'simulation' logger and other loggers in a human-readable format. Useful for monitoring simulation progress and critical errors.
*   **TensorBoard Logs (e.g., `logs/run_YYYYMMDD_HHMMSS/`):** Contains scalar metrics viewable with TensorBoard for high-level trend analysis.

This centralized approach ensures that most critical simulation events and errors are captured in a structured, analyzable format within the JSONL file, while TensorBoard provides high-level visualization and the console offers immediate feedback.
