"""
The :mod:`Diagnostic` module concerns itself with processing
and presentation of diagnostic messages.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

class Diagnostic:
    """
    A diagnostic message highlighting one or more locations
    in a single source buffer.

    :ivar level: (one of ``LEVELS``) severity level
    :ivar reason: (format string) diagnostic message
    :ivar arguments: (dictionary) substitutions for ``reason``
    :ivar location: (:class:`pyparser.source.Range`) most specific
        location of the problem
    :ivar highlights: (list of :class:`pyparser.source.Range`)
        secondary locations related to the problem that are
        likely to be on the same line
    :ivar notes: (list of :class:`Diagnostic`)
        secondary diagnostics highlighting relevant source
        locations that are unlikely to be on the same line
    """

    LEVELS = ['note', 'warning', 'error', 'fatal']
    """
    Available diagnostic levels:
        * ``fatal`` indicates an unrecoverable error.
        * ``error`` indicates an error that leaves a possibility of
          processing more code, e.g. a recoverable parsing error.
        * ``warning`` indicates a potential problem.
        * ``note`` level diagnostics do not appear by itself,
          but are attached to other diagnostics to refer to
          and describe secondary source locations.
    """

    def __init__(self, level, reason, arguments, location,
                 highlights=[], notes=[]):
        if level not in self.LEVELS:
            raise ValueError("level must be one of Diagnostic.LEVELS")

        if len(set(map(lambda x: x.source_buffer,
                       [location] + highlights))) > 1:
            raise ValueError("location and highlights must refer to the same source buffer")

        self.level, self.reason, self.arguments = \
            level, reason, arguments
        self.location, self.highlights, self.notes = \
            location, highlights, notes

    def message(self):
        """
        Returns the formatted message.
        """
        return self.reason.format(**self.arguments)

    def render(self):
        """
        Returns the human-readable location of the diagnostic in the source,
        the formatted message, the source line corresponding
        to ``location`` and a line emphasizing the problematic
        locations in the source line using ASCII art, as a list of lines.

        For example: ::

            <input>:1:8: error: cannot add integer and string
            x + (1 + "a")
                 ~ ^ ~~~
        """
        source_line = self.location.source_line().rstrip(u"\n")
        highlight_line = bytearray(u" ", 'utf-8') * len(source_line)

        for hilight in self.highlights:
            lft, rgt = hilight.column_range()
            highlight_line[lft:rgt] = bytearray(u"~", 'utf-8') * hilight.size()

        lft, rgt = self.location.column_range()
        highlight_line[lft:rgt] = bytearray(u"^", 'utf-8') * self.location.size()

        return [
            u"%s: %s: %s" % (str(self.location), self.level, self.message()),
            source_line,
            highlight_line.decode('utf-8')
        ]


class DiagnosticException(Exception):
    """
    :class:`Exception` is an exception which carries a :class:`Diagnostic`.

    :ivar diagnostic: (:class:`Diagnostic`) the diagnostic
    """
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic

    def __str__(self):
        return "\n".join(self.diagnostic.render() +
                         reduce(list.__add__, map(Diagnostic.render, self.diagnostic.notes), []))
