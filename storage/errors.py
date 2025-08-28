import functools
import inspect
import logging
import time


class DatabaseError(Exception):
    """Raised for database operation errors."""

    pass


class ExternalServiceError(Exception):
    """Raised for external API/service errors."""

    pass


class SafeOperationError(Exception):
    """Generic error for safe operation wrappers."""

    pass


def safe_db_operation(fn):
    """Decorator to wrap DB operations with logging and error escalation."""
    if inspect.iscoroutinefunction(fn):
        # Handle async functions
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                logging.error(f"[DB ERROR] {fn.__name__}: {e}", exc_info=True)
                raise DatabaseError(
                    f"Database operation failed in {fn.__name__}: {e}"
                ) from e

        return async_wrapper
    else:
        # Handle sync functions
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logging.error(f"[DB ERROR] {fn.__name__}: {e}", exc_info=True)
                raise DatabaseError(
                    f"Database operation failed in {fn.__name__}: {e}"
                ) from e

        return sync_wrapper


def safe_api_call(fn):
    """Decorator to wrap external API calls with logging and error escalation."""
    if inspect.iscoroutinefunction(fn):
        # Handle async functions
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                logging.error(f"[API ERROR] {fn.__name__}: {e}", exc_info=True)
                raise ExternalServiceError(
                    f"External API call failed in {fn.__name__}: {e}"
                ) from e

        return async_wrapper
    else:
        # Handle sync functions
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logging.error(f"[API ERROR] {fn.__name__}: {e}", exc_info=True)
                raise ExternalServiceError(
                    f"External API call failed in {fn.__name__}: {e}"
                ) from e

        return sync_wrapper
