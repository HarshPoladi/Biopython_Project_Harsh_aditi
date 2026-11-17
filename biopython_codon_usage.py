# BIOPYTHON PROJECT UPDATE 2
# TEAM/GROUP NO - 7
# Date: 26-09-2026
# Se25ubit018 Aditi Shrinath Rao
# Se25ubit034 Harsh Poladi

# E. coli Codon Usage vs tRNA Abundance
#
# Sequence source: Ensembl Bacteria, E. coli K-12 MG1655,
# assembly ASM584v2 / GCA_000005845.
# The script downloads the CDS FASTA directly.
# tRNA source: published E. coli tRNA abundance data expressed as molecules per cell.
# Codon-level match tRNA abundance is calculated by summing tRNA species
# that recognize the same codon.
#
# Install dependencies before running:
# pip install biopython scipy pandas matplotlib

from Bio import SeqIO
from collections import Counter
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import urllib.request
import os


# 1. Download E. coli CDS FASTA from Ensembl Bacteria

URL = (
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/release-63/"
    "fasta/bacteria_0_collection/"
    "escherichia_coli_str_k_12_substr_mg1655_gca_000005845/"
    "cds/"
    "Escherichia_coli_str_k_12_substr_mg1655_gca_000005845."
    "ASM584v2.cds.all.fa.gz"
)

gz_file = "ecoli_cds.fa.gz"
fa_file = "ecoli_cds.fa"

urllib.request.urlretrieve(URL, gz_file)

with gzip.open(gz_file, "rt") as f_in, open(fa_file, "w") as f_out:
    f_out.write(f_in.read())


# 2. Check the FASTA file

if not os.path.exists(fa_file):
    raise FileNotFoundError("ERROR: FASTA file was not downloaded.")

first = next(SeqIO.parse(fa_file, "fasta"))


# 3. Read CDS and count codons

cds = list(SeqIO.parse(fa_file, "fasta"))

if len(cds) == 0:
    raise ValueError("ERROR: No CDS sequences found.")

codon_counts = Counter()
stop_codons = {"TAA", "TAG", "TGA"}

for rec in cds:
    seq = str(rec.seq).upper()
    usable = len(seq) - (len(seq) % 3)

    for i in range(0, usable, 3):
        codon = seq[i:i + 3]

        if codon in stop_codons:
            continue

        if all(base in "ATGC" for base in codon):
            codon_counts[codon] += 1

if len(codon_counts) < 2:
    raise ValueError("ERROR: Too few codons were counted.")


# 4. Calculate codon frequency

total = sum(codon_counts.values())

codon_df = pd.DataFrame({
    "Codon": list(codon_counts.keys()),
    "Count": list(codon_counts.values())
})

codon_df["Frequency (%)"] = codon_df["Count"] / total * 100
codon_df["Frequency per 1000"] = codon_df["Count"] / total * 1000
codon_df = codon_df.sort_values("Codon").reset_index(drop=True)


# 5. Published tRNA abundance data
# Abundance is expressed as molecules per cell.
# Match codons are written using RNA bases (U rather than T).

trna_rows = [
    ("Ala1", 3250, "GCU,GCA,GCG"),
    ("Ala2", 617, "GCC"),
    ("Arg2", 4752, "CGU,CGC,CGA"),
    ("Arg3", 639, "CGG"),
    ("Arg4", 867, "AGA"),
    ("Arg5", 420, "AGG"),
    ("Asn", 1193, "AAC,AAU"),
    ("Asp1", 2396, "GAC,GAU"),
    ("Cys", 1587, "UGC,UGU"),
    ("Gln1", 764, "CAA"),
    ("Gln2", 881, "CAG"),
    ("Glu2", 4717, "GAA,GAG"),
    ("Gly1", 1068.5, "GGG"),
    ("Gly2", 1068.5, "GGA,GGG"),
    ("Gly3", 4359, "GGC,GGU"),
    ("His", 639, "CAC,CAU"),
    ("Ile1", 1737, "AUC,AUU"),
    ("Ile2", 1737, "AUA"),
    ("Leu1", 4470, "CUG"),
    ("Leu2", 943, "CUC,CUU"),
    ("Leu3", 666, "CUA,CUG"),
    ("Leu4", 1913, "UUG"),
    ("Leu5", 1031, "UUA,UUG"),
    ("Lys", 1924, "AAA,AAG"),
    ("Met_f1", 1211, "AUG"),
    ("Met_f2", 715, "AUG"),
    ("Met_m", 706, "AUG"),
    ("Phe", 1037, "UUC,UUU"),
    ("Pro1", 900, "CCG"),
    ("Pro2", 720, "CCC,CCU"),
    ("Pro3", 581, "CCA,CCU,CCG"),
    ("Ser1", 1296, "UCA,UCU,UCG"),
    ("Ser2", 344, "UCG"),
    ("Ser3", 1408, "AGC,AGU"),
    ("Ser5", 764, "UCC,UCU"),
    ("Thr1", 104, "ACC,ACU"),
    ("Thr2", 541, "ACG"),
    ("Thr3", 1095, "ACC,ACU"),
    ("Thr4", 916, "ACA,ACU,ACG"),
    ("Trp", 943, "UGG"),
    ("Tyr1", 769, "UAC,UAU"),
    ("Tyr2", 1261, "UAC,UAU"),
    ("Val1", 3840, "GUA,GUG,GUU"),
    ("Val2A", 630, "GUC,GUU"),
    ("Val2B", 635, "GUC,GUU")
]

trna = pd.DataFrame(
    trna_rows,
    columns=["tRNA", "Molecules_per_cell", "Match_codons_RNA"]
)

trna["Match_codons_DNA"] = trna["Match_codons_RNA"].str.replace("U", "T")


# 6. Convert tRNA species to codon-level abundance

pairs = []

for _, row in trna.iterrows():
    for codon in row["Match_codons_DNA"].split(","):
        pairs.append({
            "Codon": codon,
            "Molecules_per_cell": row["Molecules_per_cell"]
        })

pairs = pd.DataFrame(pairs)

trna_by_codon = (
    pairs.groupby("Codon", as_index=False)["Molecules_per_cell"]
    .sum()
    .rename(columns={
        "Molecules_per_cell": "Match tRNA (molecules/cell)"
    })
)


# 7. Check codon matching

matches = set(codon_df["Codon"]) & set(trna_by_codon["Codon"])

if len(matches) < 2:
    raise ValueError("STOP: Pearson correlation requires at least two matching codons.")


# 8. Merge datasets

combined = pd.merge(
    codon_df,
    trna_by_codon,
    on="Codon",
    how="inner"
)

if len(combined) < 2:
    raise ValueError("ERROR: Need at least two matching codons.")


# 9. Pearson correlation

r, p = pearsonr(
    combined["Frequency (%)"],
    combined["Match tRNA (molecules/cell)"]
)


# 10. Plot E. coli codon usage vs tRNA abundance

plt.figure(figsize=(9, 6))

plt.scatter(
    combined["Match tRNA (molecules/cell)"],
    combined["Frequency (%)"]
)

for _, row in combined.iterrows():
    plt.annotate(
        row["Codon"],
        (
            row["Match tRNA (molecules/cell)"],
            row["Frequency (%)"]
        ),
        fontsize=7
    )

plt.xlabel("Match tRNA Abundance (molecules/cell)")
plt.ylabel("Codon frequency (%)")
plt.title("E. coli Codon Usage vs tRNA Abundance")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "E_coli_codon_vs_tRNA.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
