from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from datasets import load_dataset
import pandas as pd
from collections import Counter
import random
import time


def find_majority(nums):
    counts = Counter(nums)
    majority_count = len(nums) // 2
    for num, count in counts.items():
        if count > majority_count:
            return num
    return random.choice(nums) if nums else None


Entrez.email = "skuthpadi@umass.edu"

def blast_search(sequence):
    print("Submitting BLAST search")
    result_handle = NCBIWWW.qblast("blastn", "nt", sequence, format_type="XML")
    print("BLAST search complete.")
    return result_handle

def parse_blast_results(result_handle, evalue_threshold=1e-3):
    print("Parsing BLAST results")
    blast_record = NCBIXML.read(result_handle)
    accession_ids = []
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            if hsp.expect < evalue_threshold:
                accession_ids.append(alignment.accession)
                break
    print(f"Found {len(accession_ids)} significant BLAST hit(s).")
    return accession_ids

def extract_ec_numbers(accession_ids):
    ec_numbers = []
    for acc in accession_ids:
        print(f"Fetching GenBank record for accession {acc}...")
        try:
            # Fetching the GenBank record
            # using 'nucleotide' db and rettype 'gb'
            handle = Entrez.efetch(db="nucleotide", id=acc, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "genbank")
            handle.close()
        except Exception as e:
            print(f"Error fetching accession {acc}: {e}")
            continue
        
        for feature in record.features:
            if feature.type == "CDS":
                if "EC_number" in feature.qualifiers:
                    for ec in feature.qualifiers["EC_number"]:
                        ec_numbers.append(ec)
    return ec_numbers

def predict_ec_numbers_from_sequence(sequence):

    blast_handle = blast_search(sequence)
    accession_ids = parse_blast_results(blast_handle)
    ec_numbers = extract_ec_numbers(accession_ids)
    return ec_numbers

if __name__ == "__main__":
    ds = load_dataset("tattabio/ec_classification_dna")['test']
    results = []

    try:
        existing_df = pd.read_csv("predicted_ec_results.csv")
        processed_entries = set(existing_df['Entry'].tolist())
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        processed_entries = set()


    for i, example in enumerate(ds):
        sequence = example['Sequence']

        if example['Entry'] in processed_entries:
            print(f"Skipping already processed entry: {example['Entry']}")
            continue
        
        print("Running enzyme prediction pipeline...")
        predicted_ec_numbers = predict_ec_numbers_from_sequence(sequence)
        
        if predicted_ec_numbers:
            print("\nPredicted Enzyme Commission (EC) Number(s):")
            for ec in predicted_ec_numbers:
                print(f"  - {ec}")
        else:
            print("No EC numbers were predicted from the input sequence.")
        
        result = {
            "Entry": example['Entry'],
            "Sequence": sequence,
            "Predicted_EC_Numbers": find_majority(list(predicted_ec_numbers))
        }

        result_df = pd.DataFrame([result])
        result_df.to_csv("predicted_ec_results.csv", mode='a', header=not existing_df.shape[0], index=False)

        print("Saved to file.")
        time.sleep(10)



