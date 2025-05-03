from Bio.Blast import NCBIWWW, NCBIXML
import time

def run_blastn(sequence, email="skuthpadi@umass.edu", program="blastn", database="nt", hitlist_size=1):
    NCBIWWW.email = email
    
    print(f"Running BLAST for sequence (first 30 nt): {sequence[:30]}...")
    result_handle = NCBIWWW.qblast(program, database, sequence, hitlist_size=hitlist_size)
    
    blast_record = NCBIXML.read(result_handle)
    if blast_record.alignments:
        top_hit = blast_record.alignments[0]
        title = top_hit.title
        print(f"Top hit title: {title}")
        return title
    else:
        print("No hits found.")
        return None

example_seq = "ATGCGTACGTAGCTAGCGTACGATCGTAGCTAGCATCGATG"

top_hit_description = run_blastn(example_seq)