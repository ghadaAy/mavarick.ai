import traceback

from result import Err


class CustomError(Err):
    def __init__(self, message: str, exc: Exception) -> None:
        self.formatted_traceback = "".join(traceback.format_tb(exc.__traceback__))
        self.message = message
        super().__init__(f"{self.message}:{self.formatted_traceback}")

    def __str__(self) -> str:
        return self.formatted_traceback

    def unwrap_err(self) -> str:
        return str(self)
