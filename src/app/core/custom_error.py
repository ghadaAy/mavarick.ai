"""Module for custom error handling with traceback logging."""
import traceback

from result import Err


class CustomError(Err):
    """Error class that logs message and traceback."""

    def __init__(self, message: str, exc: Exception) -> None:
        """
        Initializes the error with a message and formatted traceback.

        Args:
            message (str): Custom error message.
            exc (Exception): Original exception to extract traceback from.

        """
        self.formatted_traceback = "".join(traceback.format_tb(exc.__traceback__))
        self.message = message
        super().__init__(f"{self.message}:{self.formatted_traceback}")

    def __str__(self) -> str:
        """Returns the formatted traceback as a string representation of the error."""
        return self.formatted_traceback

    def unwrap_err(self) -> str:
        """Unwraps and returns the error message with traceback details."""
        return str(self)
