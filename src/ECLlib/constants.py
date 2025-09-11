from ECLlib import unfmt_block

# Empty block that terminates a SEQNUM - ENDSOL section in UNRST-files
ENDSOL = unfmt_block.from_data('ENDSOL', [], 'mess')
