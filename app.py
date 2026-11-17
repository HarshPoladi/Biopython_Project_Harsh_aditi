import gzip
import os
import urllib.request
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from Bio import SeqIO
from scipy.stats import pearsonr

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Codon–tRNA Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A simple visual theme using CSS
st.markdown(
    """
    <style>
    /* A quieter, more natural layout with clear contrast */
    .stApp { background: #fbfbf8; color: #263238; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1120px; }
    html, body, [class*="css"] { color: #263238; }
    p, li, label, .stMarkdown { color: #34434a; }
    h1, h2, h3, h4 { color: #20343b !important; letter-spacing: -0.02em; }
    .hero {
        padding: 1.5rem 1.7rem;
        border-left: 5px solid #52796f;
        border-radius: 5px;
        background: #eef3ef;
        color: #20343b;
        margin-bottom: 1.6rem;
    }
    .hero h1 { color: #20343b !important; font-size: 2rem; margin-bottom: .4rem; }
    .hero p { color: #52636a !important; font-size: 1rem; margin-bottom: 0; }
    .section-note {
        background: #f1f4f1; border: 1px solid #dfe7e1; border-radius: 6px;
        padding: 1rem 1.2rem; margin: .5rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e1e6e2;
        padding: 1rem; border-radius: 6px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #263b40 !important;
    }
    div.stButton > button[kind="primary"] {
        background: #52796f; border: 1px solid #52796f;
        border-radius: 6px; padding: .6rem 1rem; font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #3f6259; border-color: #3f6259;
    }
    [data-testid="stSidebar"] { background: #f1f4f1; }
    [data-testid="stTabs"] button { color: #34434a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Codon usage and tRNA abundance</h1>
      <p>A small bioinformatics study using coding sequences from <i>Escherichia coli</i> K-12 MG1655.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# DATA SOURCES
# ---------------------------------------------------------
URL = (
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/release-63/"
    "fasta/bacteria_0_collection/"
    "escherichia_coli_str_k_12_substr_mg1655_gca_000005845/"
    "cds/"
    "Escherichia_coli_str_k_12_substr_mg1655_gca_000005845."
    "ASM584v2.cds.all.fa.gz"
)

# tRNA abundance values and codon matches used in the project dataset.
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

# ---------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_and_count_codons():
    compressed_file = "ecoli_cds.fa.gz"
    fasta_file = "ecoli_cds.fa"

    if not os.path.exists(fasta_file):
        urllib.request.urlretrieve(URL, compressed_file)
        with gzip.open(compressed_file, "rt") as source, open(fasta_file, "w") as target:
            target.write(source.read())

    counts = Counter()
    stop_codons = {"TAA", "TAG", "TGA"}

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        usable_length = len(sequence) - (len(sequence) % 3)

        for position in range(0, usable_length, 3):
            codon = sequence[position:position + 3]
            if codon not in stop_codons and all(base in "ATGC" for base in codon):
                counts[codon] += 1

    if not counts:
        raise ValueError("No valid codons were found in the downloaded sequence data.")

    total = sum(counts.values())
    codon_table = pd.DataFrame(
        {"Codon": list(counts.keys()), "Count": list(counts.values())}
    )
    codon_table["Frequency (%)"] = codon_table["Count"] / total * 100
    codon_table["Frequency per 1000"] = codon_table["Count"] / total * 1000
    return codon_table.sort_values("Codon").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_trna_tables():
    trna_table = pd.DataFrame(
        TRNA_ROWS,
        columns=["tRNA", "Molecules per cell", "Recognized RNA codons"],
    )
    trna_table["Recognized DNA codons"] = trna_table["Recognized RNA codons"].str.replace(
        "U", "T", regex=False
    )

    codon_matches = []
    for _, row in trna_table.iterrows():
        for codon in row["Recognized DNA codons"].split(","):
            codon_matches.append(
                {"Codon": codon, "Molecules per cell": row["Molecules per cell"]}
            )

    codon_trna = (
        pd.DataFrame(codon_matches)
        .groupby("Codon", as_index=False)["Molecules per cell"]
        .sum()
        .rename(columns={"Molecules per cell": "Matching tRNA abundance"})
    )
    return trna_table, codon_trna


def run_analysis():
    codon_table = download_and_count_codons()
    trna_table, codon_trna = build_trna_tables()
    combined = pd.merge(codon_table, codon_trna, on="Codon", how="inner")

    if len(combined) < 2:
        raise ValueError("There are not enough matching codons to calculate correlation.")

    r_value, p_value = pearsonr(
        combined["Frequency (%)"],
        combined["Matching tRNA abundance"],
    )
    return codon_table, trna_table, combined, r_value, p_value


# ---------------------------------------------------------
# SIDEBAR / PROJECT GUIDE
# ---------------------------------------------------------
with st.sidebar:
    st.header("Project guide")
    st.write("**Organism:** E. coli K-12 MG1655")
    st.write("**Main question:** Is codon usage associated with the abundance of matching tRNAs?")
    st.divider()
    st.subheader("How to use")
    st.write("1. Click **Run analysis**.")
    st.write("2. Wait while the coding sequences are downloaded and counted.")
    st.write("3. Explore the correlation, graph, and tables.")
    st.divider()
    st.caption("Sequence source: Ensembl Bacteria, ASM584v2 CDS FASTA.")
    st.caption("tRNA values and codon matches: project dataset.")

# ---------------------------------------------------------
# INTRODUCTION
# ---------------------------------------------------------
intro_left, intro_right = st.columns([1.35, 1])
with intro_left:
    st.subheader("Project overview")
    st.write(
        "A codon is a group of three DNA or RNA bases that corresponds to an "
        "amino acid or a stop signal. During protein synthesis, tRNA molecules "
        "recognize codons and bring the corresponding amino acids to the "
        "ribosome."
    )
    st.write(
        "This project counts codons in E. coli coding sequences and compares "
        "their frequencies with the abundance of tRNAs listed as recognizing "
        "those codons."
    )
with intro_right:
    st.markdown(
        """
        <div class="section-note">
          <b>What the analysis measures</b><br><br>
          <b>Codon usage</b><br>
          How often each codon appears in the coding sequences.<br><br>
          <b>tRNA abundance</b><br>
          The summed abundance of tRNAs matched to each codon.<br><br>
          <b>Correlation</b><br>
          Whether the two measurements tend to increase or decrease together.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Run the analysis")
st.write("The first run downloads the E. coli coding-sequence file. Later runs use cached results.")

if st.button("▶  Run analysis", type="primary", use_container_width=True):
    try:
        with st.spinner("Downloading sequence data and calculating codon frequencies..."):
            results = run_analysis()
        st.session_state["analysis_results"] = results
        st.success("Analysis completed.")
    except Exception as error:
        st.error(f"The analysis could not be completed: {error}")

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
if "analysis_results" in st.session_state:
    codon_table, trna_table, combined, r_value, p_value = st.session_state["analysis_results"]

    st.divider()
    st.subheader("Analysis overview")
    st.write("These values summarize the relationship calculated from the matched codons.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Matched codons", f"{len(combined)}")
    m2.metric("Pearson correlation (r)", f"{r_value:.3f}")
    m3.metric("p-value", f"{p_value:.3g}")

    if r_value > 0:
        direction = "positive"
    elif r_value < 0:
        direction = "negative"
    else:
        direction = "no"
    st.info(
        f"The calculated Pearson correlation is {direction} "
        f"(r = {r_value:.3f}). The p-value is {p_value:.3g}. "
        "Correlation describes an association; it does not prove that one factor causes the other."
    )

    tab_graph, tab_data, tab_method = st.tabs(
        ["📈 Correlation graph", "📋 Data tables", "ℹ️ Method"]
    )

    with tab_graph:
        st.subheader("Codon frequency vs matching tRNA abundance")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.scatter(
            combined["Matching tRNA abundance"],
            combined["Frequency (%)"],
            alpha=0.8,
        )
        for _, row in combined.iterrows():
            ax.annotate(
                row["Codon"],
                (row["Matching tRNA abundance"], row["Frequency (%)"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xlabel("Matching tRNA abundance (molecules per cell)")
        ax.set_ylabel("Codon frequency (%)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Each point represents one codon. Its position compares codon frequency with the abundance of matching tRNA molecules.")

    with tab_data:
        st.subheader("Combined results")
        st.dataframe(combined, use_container_width=True, hide_index=True)
        with st.expander("Codon frequency table"):
            st.dataframe(codon_table, use_container_width=True, hide_index=True)
        with st.expander("tRNA abundance and codon matches"):
            st.dataframe(trna_table, use_container_width=True, hide_index=True)

        csv_bytes = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download combined results (CSV)",
            data=csv_bytes,
            file_name="ecoli_codon_trna_results.csv",
            mime="text/csv",
        )

    with tab_method:
        st.subheader("How the analysis works")
        st.markdown(
            """
            1. **Download sequences:** the app retrieves the E. coli K-12 MG1655 coding-sequence FASTA file.
            2. **Count codons:** each coding sequence is read in groups of three bases. Stop codons and incomplete/ambiguous codons are excluded.
            3. **Calculate frequency:** each codon's count is divided by the total number of counted codons.
            4. **Match tRNAs:** the project tRNA table links tRNA species to the codons they recognize. Abundances for tRNAs matched to the same codon are added together.
            5. **Calculate correlation:** Pearson's correlation is calculated between codon frequency and matching tRNA abundance.
            """
        )

st.divider()
st.caption(
    "Student bioinformatics project · Results depend on the sequence file and the tRNA abundance/matching dataset used."
)
