"""This module contains bases classes used for indexing purposes."""

from abc import ABCMeta, abstractmethod

from result import Result

from app.core.custom_error import AppError


class IStrategy[T](metaclass=ABCMeta):
    """Strategy interface."""

    @abstractmethod
    async def execute(self) -> Result[T, AppError]:
        """
        Executes the given strategy.

        Returns:
            Result[T,AppError]: wrapped result or error.

        """


class IFileReader(IStrategy[list[str]]):
    """A simple file reader interface."""

    def __init__(self, file_path: str) -> None:
        """
        Args:
            file_path (str): path to the file to read.

        """
        self.file_path = file_path
