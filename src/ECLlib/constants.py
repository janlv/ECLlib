from ECLlib import unfmt_block

DEBUG = False
ENDIAN = '>'  # Big-endian
ECL2IX_LOG = 'ecl2ix.log'

# Empty block that terminates a SEQNUM - ENDSOL section in UNRST-files
ENDSOL = unfmt_block.from_data('ENDSOL', [], 'mess')
