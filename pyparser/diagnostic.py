"""
The :mod:`Diagnostic` module concerns itself with processing
and presentation of diagnostic messages.
"""

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
            raise ValueError, "level must be one of Diagnostic.LEVELS"

        if len(set(map(lambda x: x.source_buffer,
                       [location] + highlights))) > 1:
            raise ValueError, "location and highlights must refer to the same source buffer"

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
        """
        source_line = self.location.source_line().rstrip(u"\n")
        highlight_line = bytearray(' ') * len(source_line)

        for hilight in self.highlights:
            lft, rgt = hilight.column_range()
            highlight_line[lft:rgt] = bytearray('~') * hilight.size()

        lft, rgt = self.location.column_range()
        highlight_line[lft:rgt] = bytearray('^') * self.location.size()

        return [
            "%s: %s: %s" % (str(self.location), self.level, self.message()),
            source_line,
            unicode(highlight_line)
        ]
