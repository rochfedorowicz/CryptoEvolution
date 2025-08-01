# utils/stdout_redirecting.py

# global import
import logging
import re
import sys
from contextlib import contextmanager
from typing import Any

# local import

class LoggingOut():
    """
    Implements a custom output stream that redirects stdout to the logging module.
    It captures text written to stdout and logs it as info messages.
    """

    # local constants
    __NEW_LINE: str = '\n'

    def __init__(self, filter: bool = True) -> None:
        """
        Class constructor. Initializes an empty text buffer to capture stdout.
        """

        self.__text_buffer: str = ''
        self.__characters_to_be_filtered = re.compile(r'\x08+|\r') if filter else None

    def write(self, text) -> None:
        """
        Writes text to the internal buffer.

        Parameters:
            text (str): The text to write to the buffer.
        """

        if self.__characters_to_be_filtered is not None:
            text = self.__characters_to_be_filtered.sub('', text)

        self.__text_buffer += text

    def flush(self) -> None:
        """
        Flushes the internal buffer by logging all captured text.
        If the buffer contains new lines, it splits the text into lines and logs each line.
        """

        if self.__NEW_LINE in self.__text_buffer:
            lines = self.__text_buffer.split(self.__NEW_LINE)
            for line in lines:
                if line.strip():
                    logging.info(line)
        self.__text_buffer = ''

@contextmanager
def redirect_stdout_to_logging(filter: bool = True) -> Any:
    """
    Context manager that redirects stdout to a custom logging stream.
    """

    original_stdout = sys.stdout
    logging_stream_stdout = LoggingOut(filter = filter)

    try:
        sys.stdout = logging_stream_stdout
        yield
    finally:
        sys.stdout = original_stdout
        logging_stream_stdout.flush()