class FrealignParFileBase(object):
    COMMENT_TOKEN = 'C'
    COLUMN_SEPARATOR = ' '

    def __init__(self, path, mode, columns):
        self._file = open(path, mode)
        self._columns = columns

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        """ Convert a line into a dict with _header as keys.
        :return: yield a dict - single row
        """
        for line in self._file:
            line = line.strip()
            if not line.startswith(self.COMMENT_TOKEN): # Skip comments
                yield self._parseLine(line)

    def writeRow(self, row):
        if isinstance(row, list):
            # Ensure it has the correct size
            if len(row) != len(self._columns):
                raise ValueError('The row has an invalid amount of columns')

            # Format the row
            row = [column['format'].format(value) for (column, value) in zip(self._columns, row)]

            # Merge all the columns into a line
            line = self.COLUMN_SEPARATOR.join(row) + '\n'
            self._file.write(line)

        elif isinstance(row, dict):
            # Recursively call to itself to reach the case above
            row = [row[column['name']] for column in self._columns]
            self.writeRow(row)

        else:
            raise TypeError('Invalid type for row')

    def writeComment(self, comment):
        line = self.COMMENT_TOKEN + self.COLUMN_SEPARATOR + comment + '\n'
        self._file.write(line)

    def writeHeader(self):
        headers = [c['header'] for c in self._columns]
        comment = self.COLUMN_SEPARATOR.join(headers)
        self.writeComment(comment)

    def close(self):
        self._file.close()

    def _parseLine(self, line):
        # Obtain the key and tokens arrays
        keys = [c['name'] for c in self._columns]
        tokens = line.split()

        # Ensure that keys and tokens are matched
        if len(keys) != len(tokens):
            raise ValueError('Invalid line read: '+line)

        # Type the tokens to obtain the values
        values = [column['type'](token) for (column, token) in zip(self._columns, tokens)]

        # All ok, build a dictionary with the key-value pairs
        return dict(zip(keys, values))


class MinimalFrealignParFile(FrealignParFileBase):
    COLUMNS = [
        { 'name': 'mic_id',         'header': '    ',       'format': '{:<6d}',     'type': int},
        { 'name': 'defocus_u',      'header': 'DF1     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'defocus_v',      'header': 'DF2     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'defocus_angle',  'header': 'ANGAST  ',   'format': '{:<8.2f}',   'type': float},
    ]

    def __init__(self, path, mode):
        FrealignParFileBase.__init__(self, path, mode, self.COLUMNS)

    def writeParticle(self, particle):
        micId = particle.getMicId()
        ctfModel = particle.getCTF()
        
        # Ensure that a micrograph id is given
        if micId is None:
            micId = 1

        # Determine the CTF data
        (dfU, dfV, dfAngle) = ctfModel.getDefocus() if ctfModel is not None else (0, 0, 0)

        # Write values in order
        row = [
            micId,
            dfU,
            dfV,
            dfAngle
        ]
        self.writeRow(row)

    def writeParticleSet(self, particleSet):
        for particle in particleSet:
            self.writeParticle(particle)

    def writeHeaderAndParticleSet(self, particleSet):
        self.writeHeader()
        self.writeParticleSet(particleSet)


class FullFrealignParFile(FrealignParFileBase):
    COLUMNS = [
        { 'name': 'mic_id',         'header': '    ',       'format': '{:<6d}',     'type': int},
        { 'name': 'psi',            'header': 'PSI     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'theta',          'header': 'THETA   ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'phi',            'header': 'PHI     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'shift_x',        'header': 'SHX     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'shift_y',        'header': 'SHY     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'magnification',  'header': 'MAG     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'film',           'header': 'FILM',       'format': '{:<4d}',     'type': int},
        { 'name': 'defocus_u',      'header': 'DF1     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'defocus_v',      'header': 'DF2     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'defocus_angle',  'header': 'ANGAST  ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'p_shift',        'header': 'PSHIFT  ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'occupancy',      'header': 'OCC     ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'log_p',          'header': 'LOGP    ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'sigma',          'header': 'SIGMA   ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'score',          'header': 'SCORE   ',   'format': '{:<8.2f}',   'type': float},
        { 'name': 'change',         'header': 'CHANGE  ',   'format': '{:<8.2f}',   'type': float},
    ]

    def __init__(self, path, mode):
        FrealignParFileBase.__init__(self, path, mode, self.COLUMNS)

    def writeParticle(self, particle, id=None, occupancy=None, logP=None, sigma=None):
        from .convert import geometryFromMatrix

        micId = id if id is not None else particle.getMicId()
        ctfModel = particle.getCTF()
        acquisition = particle.getAcquisition()
        transform = particle.getTransform()
        
        
        # Ensure that a micrograph id is given
        if micId is None:
            micId = 1

        # Determine the geometry information
        (shifts, angles) = geometryFromMatrix(transform.getMatrix()) if transform is not None else ((0, 0, 0), (0, 0, 0))
        (psi, theta, phi) = angles
        (shX, shY, _) = shifts

        # Determine the acquisition information
        mag = acquisition.getMagnification() if acquisition is not None else 0
        film = 1

        # Determine the CTF data
        (dfU, dfV, dfAngle) = ctfModel.getDefocus() if ctfModel is not None else (0, 0, 0)

        # Initialize the refinement data
        if occupancy is None:
            occupancy = 100
        if logP is None:
            logP = 0
        if sigma is None:
            sigma = 0.5
        pShift = 0
        score = 0
        change = 0

        row = [
            micId,
            psi,
            theta,
            phi,
            shX,
            shY,
            mag,
            film,
            dfU,
            dfV,
            dfAngle,
            pShift,
            occupancy,
            logP,
            sigma,
            score,
            change
        ]
        self.writeRow(row)

    def writeParticleSet(self, particleSet):
        for particle in particleSet:
            self.writeParticle(particle)

    def writeHeaderAndParticleSet(self, particleSet):
        self.writeHeader()
        self.writeParticleSet(particleSet)

class FrealignStatisticsFile(FrealignParFileBase):
    COLUMNS = [
        { 'name': 'shell',          'header': 'SHELL ',         'format': '{:<8d}',     'type': int},
        { 'name': 'resolution',     'header': 'RESOLUTION    ', 'format': '{:<14.5f}',  'type': float},
        { 'name': 'ring_radius',    'header': 'RING_RADIUS   ', 'format': '{:<14.5f}',  'type': float},
        { 'name': 'fsc',            'header': 'FSC           ', 'format': '{:<14.5f}',  'type': float},
        { 'name': 'part_fsc',       'header': 'PART_FSC      ', 'format': '{:<14.5f}',  'type': float},
        { 'name': 'part_sqrt_ssnr', 'header': 'PART_SSNR_SQRT', 'format': '{:<14.5f}',  'type': float},
        { 'name': 'rec_sqrt_ssnr',  'header': 'REC_SSNR_SQRT ', 'format': '{:<14.5f}',  'type': float},
    ]

    def __init__(self, path, mode):
        FrealignParFileBase.__init__(self, path, mode, self.COLUMNS)
