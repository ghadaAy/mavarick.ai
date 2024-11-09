"""Module for custom error handling with traceback logging."""

import os
import traceback

from pydantic import ValidationError
from result import Err


class AppError(Err):
    """Error class that logs message and traceback."""

    def __init__(self, message: str, exc: Exception | None = None) -> None:
        """
        Initializes the error with a message and formatted traceback.

        Args:
            message (str): Custom error message.
            exc (Exception): Original exception to extract traceback from.

        """
        self.message = message
        self.traceback = self.format_exception(exc) if exc else ""
        super().__init__(f"{self.message}:{self.traceback}")

    def __str__(self) -> str:
        """Returns the formatted traceback as a string representation of the error."""
        return self.traceback

    def format_exception(self, exc: BaseException) -> str:
        """
        Set a properly formatted exception traceback on the error object.

        Args:
            exc (BaseException): the exception that ocurred.

        Returns:
            None

        """
        if isinstance(exc, ValidationError):
            self.message += os.linesep
            self.message += exc.json()

        return os.linesep.join(traceback.format_exception(exc))
