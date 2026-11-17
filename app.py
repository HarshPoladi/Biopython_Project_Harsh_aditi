import gzip
import os
import urllib.request
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from Bio import SeqIO
from scipy.stats import pearsonr


st.set_page_config(
    page_title="E. coli Codon Usage Analyzer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 E. coli Codon Usage vs tRNA Abundance")
st.caption("Bioinformatics project · E. coli K-12 MG1655 · ASM584v2")

st.write(
    "This app compares how frequently each codon is used in "
    "E. coli coding sequences with the abundance of tRNAs that can "
    "recognize that codon."
)

URL = (
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/release-63/"
    "fasta/bacteria_0_collection/"
    "escherichia_coli_str_k_12_substr_mg1655_gca_000005845/"
    "cds/"
    "Escherichia_coli_str_k_12_substr_mg1655_gca_000005845."
    "ASM584v2.cds.all.fa.gz"
)

TRNA_ROWS = [
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
    ("Val2B", 635, "GUC,GUU"),
]


@st.cache_data(show_spinner=False)
def download_and_count_codons():
    gz_file = "ecoli_cds.fa.gz"
    fa_file = "ecoli_cds.fa"

    if not os.path.exists(fa_file):
        urllib.request.urlretrieve(URL, gz_file)
        with gzip.open(gz_file, "rt") as f_in, open(fa_file, "w") as f_out:
            f_out.write(f_in.read())

    codon_counts = Counter()
    stop_codons = {"TAA", "TAG", "TGA"}

    for record in SeqIO.parse(fa_file, "fasta"):
        sequence = str(record.seq).upper()
        usable_length = len(sequence) - (len(sequence) % 3)

        for i in range(0, usable_length, 3):
            codon = sequence[i:i + 3]

            if codon in stop_codons:
                continue

            if all(base in "ATGC" for base in codon):
                codon_counts[codon] += 1

    if not codon_counts:
        raise ValueError("No valid coding-sequence codons were found.")

    total = sum(codon_counts.values())
    codon_df = pd.DataFrame(
        {
            "Codon": list(codon_counts.keys()),
            "Count": list(codon_counts.values()),
        }
    )
    codon_df["Frequency (%)"] = codon_df["Count"] / total * 100
    codon_df["Frequency per 1000"] = codon_df["Count"] / total * 1000
    return codon_df.sort_values("Codon").reset_index(drop=True)


@st.cache_data
def build_trna_table():
    trna = pd.DataFrame(
        TRNA_ROWS,
        columns=["tRNA", "Molecules_per_cell", "Match_codons_RNA"],
    )
    trna["Match_codons_DNA"] = trna["Match_codons_RNA"].str.replace(
        "U", "T", regex=False
    )

    pairs = []
    for _, row in trna.iterrows():
        for codon in row["Match_codons_DNA"].split(","):
            pairs.append(
                {
                    "Codon": codon,
                    "Molecules_per_cell": row["Molecules_per_cell"],
                }
            )

    pairs_df = pd.DataFrame(pairs)
    trna_by_codon = (
        pairs_df.groupby("Codon", as_index=False)["Molecules_per_cell"]
        .sum()
        .rename(
            columns={
                "Molecules_per_cell": "Match tRNA (molecules/cell)"
            }
        )
    )
    return trna, trna_by_codon


st.divider()

if st.button("Analyze E. coli codon usage", type="primary"):
    try:
        with st.spinner("Downloading sequence data and analyzing codons..."):
            codon_df = download_and_count_codons()
            trna_df, trna_by_codon = build_trna_table()
            combined = pd.merge(
                codon_df,
                trna_by_codon,
                on="Codon",
                how="inner",
            )

            if len(combined) < 2:
                raise ValueError(
                    "At least two matching codons are needed for correlation."
                )

            correlation, p_value = pearsonr(
                combined["Frequency (%)"],
                combined["Match tRNA (molecules/cell)"],
            )

        st.success("Analysis completed successfully.")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Codons analyzed", len(combined))
        metric2.metric("Pearson correlation (r)", f"{correlation:.3f}")
        metric3.metric("p-value", f"{p_value:.3g}")

        st.subheader("Codon usage vs tRNA abundance")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            combined["Match tRNA (molecules/cell)"],
            combined["Frequency (%)"],
        )

        for _, row in combined.iterrows():
            ax.annotate(
                row["Codon"],
                (
                    row["Match tRNA (molecules/cell)"],
                    row["Frequency (%)"],
                ),
                fontsize=7,
            )

        ax.set_xlabel("Matching tRNA abundance (molecules/cell)")
        ax.set_ylabel("Codon frequency (%)")
        ax.set_title("E. coli Codon Usage vs tRNA Abundance")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Combined codon and tRNA data")
        st.dataframe(combined, use_container_width=True, hide_index=True)

        with st.expander("View codon frequency table"):
            st.dataframe(codon_df, use_container_width=True, hide_index=True)

        with st.expander("View published tRNA abundance table"):
            st.dataframe(trna_df, use_container_width=True, hide_index=True)

        csv_data = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download analyzed data as CSV",
            data=csv_data,
            file_name="ecoli_codon_trna_analysis.csv",
            mime="text/csv",
        )

        st.caption(
            "Interpretation: Pearson's r describes the strength and direction "
            "of a linear association. It does not establish causation."
        )

    except Exception as error:
        st.error(f"Analysis could not be completed: {error}")

else:
    st.info(
        "Click **Analyze E. coli codon usage** to download the coding sequences, "
        "calculate codon frequencies, match tRNA abundance, and display the "
        "correlation and graph."
    )

st.divider()
st.caption("Biopython Project · Group 7")
