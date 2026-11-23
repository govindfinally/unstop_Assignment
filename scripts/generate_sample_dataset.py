"""Generate the 15-file synthetic multimodal dataset, external KB, and ground-truth manifest.

All texts are 100% original synthetic business case studies (zero copyrighted or private data).
Spoken MP4 clips are generated offline via Windows TTS (pyttsx3) + bundled ffmpeg (imageio-ffmpeg).

Usage:
    python scripts/generate_sample_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import docx
from fpdf import FPDF
import imageio_ffmpeg
import pyttsx3

from src.config import DEFAULT_SETTINGS
from src.logger import get_logger, setup_logging

setup_logging(DEFAULT_SETTINGS.log_dir, DEFAULT_SETTINGS.log_file_name, "INFO")
logger = get_logger("scripts.generate_dataset")

SUBMISSIONS_DIR = PROJECT_ROOT / "sample_submissions"
EXTERNAL_KB_DIR = PROJECT_ROOT / "external_knowledge"

# ~150-word shared competition prompt for Scenario 4 (Benign Common Formatting)
SHARED_CASE_PROMPT = (
    "Unstop National Strategy Case Challenge 2026 Problem Statement. "
    "Evaluate the structural cost drivers, operational bottlenecks, and unit economics facing mid-market "
    "enterprises expanding across Tier-2 and Tier-3 Indian cities over the next twenty-four months. "
    "Your team must diagnose the primary margin leakage points using realistic operational trade-offs "
    "and propose a phased implementation roadmap covering capital allocation, supplier governance, and "
    "service-level reliability targets. "
    "All recommendations must balance short-term working capital discipline against long-term customer "
    "retention and measurable return on invested capital. "
    "Submissions should clearly state key operational assumptions, highlight execution risks across "
    "regional hubs, and define quarterly key performance indicators that leadership can audit directly. "
    "Please adhere strictly to the competition integrity policy and cite any external market benchmarks "
    "referenced during your financial modeling and strategic evaluation."
)

# Page 2 passage copied verbatim into MP4 pitch (with spoken fillers)
CROSS_MODAL_PARAGRAPH = (
    "Operating margins contracted by 4.2 percent during the third quarter primarily due to port "
    "congestion and erratic raw material lead times across our western distribution corridor. "
    "To reverse this decline, we propose consolidating three regional fulfillment centers into a "
    "vertically integrated assembly hub that directly links component sourcing with final dispatch. "
    "This structural shift cuts inter-warehouse freight handoffs by thirty-eight percent and restores "
    "on-time delivery reliability above ninety-six percent within two operating quarters."
)

# Spoken version with realistic fillers ("um", "uh", "ah") for Scenario 1
CROSS_MODAL_SPOKEN_SCRIPT = (
    "Um, operating margins contracted by 4.2 percent during the third quarter, uh, primarily due to port "
    "congestion and erratic raw material lead times across our western distribution corridor. "
    "Ah, to reverse this decline, we propose consolidating three regional fulfillment centers into a "
    "vertically integrated assembly hub that directly links component sourcing with final dispatch. "
    "Um, this structural shift cuts inter-warehouse freight handoffs by thirty-eight percent and restores "
    "on-time delivery reliability above ninety-six percent within two operating quarters."
)

ORIGINAL_PITCH_SPOKEN_SCRIPT = (
    "Good afternoon judges. Our healthcare diagnostics pilot deploys portable ultrasound probes across "
    "forty rural primary clinics in Odisha using solar battery carts and asynchronous radiologist review. "
    "By batching encrypted scans over low-bandwidth satellite links overnight, we cut prenatal screening "
    "wait times from fourteen days down to eighteen hours while keeping per-patient fees under two hundred rupees."
)

# External Knowledge Base Articles (open/synthetic reference summaries)
EXTERNAL_ARTICLES: Dict[str, str] = {
    "ref_maritime_supply_chains.txt": (
        "Global maritime supply chain resilience depends on multi-port routing flexibility and buffered "
        "safety stock near high-demand consumption clusters. Historical freight studies show that "
        "single-corridor container dependency exposes importers to cascading demurrage fees whenever "
        "customs clearance slows by more than forty-eight hours. Leading logistics operators mitigate "
        "disruption risk through dual-sourcing critical sub-assemblies, real-time vessel telemetry, "
        "and dynamic reorder triggers tied to port dwell indices rather than static monthly calendars."
    ),
    "ref_microgrid_economics.txt": (
        "Decentralized solar microgrids achieve commercial viability in off-grid agricultural belts "
        "when productive anchor loads such as grain mills and cold rooms absorb daytime surplus generation. "
        "Pay-as-you-go smart meters paired with automated mobile billing reduce collection leakage below "
        "three percent while improving battery lifecycle utilization through predictable load scheduling."
    ),
    "ref_digital_payments_india.txt": (
        "Instant interoperable retail payment rails lowered merchant onboarding friction across small "
        "neighborhood stores by eliminating hardware terminal rental fees. Transaction histories created "
        "through zero-fee QR settlements now serve as verifiable cash-flow collateral for micro-credit "
        "underwriting in underserved semi-urban districts."
    ),
}


def _write_pdf(path: Path, pages: List[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, page_text)
    pdf.output(str(path))


def _write_docx(path: Path, paragraphs: List[str]) -> None:
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))


def _synthesize_spoken_mp4(path: Path, script: str) -> None:
    """Generate offline spoken-English WAV via pyttsx3 and mux into standard MP4 via ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "speech.wav"
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.save_to_file(script, str(wav_path))
        engine.runAndWait()
        engine.stop()

        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x1e293b:s=320x240:r=10",
            "-i",
            str(wav_path),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _create_silent_mp4(path: Path, duration_sec: int = 5) -> None:
    """Create a valid MP4 with a completely silent AAC audio track (Scenario 3)."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x0f172a:s=320x240:r=10:d={duration_sec}",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=16000:cl=mono:d={duration_sec}",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_all() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_KB_DIR.mkdir(parents=True, exist_ok=True)

    # 1. External Knowledge Base
    for fname, content in EXTERNAL_ARTICLES.items():
        (EXTERNAL_KB_DIR / fname).write_text(content, encoding="utf-8")
    logger.info("Wrote %d external reference article(s) to %s", len(EXTERNAL_ARTICLES), EXTERNAL_KB_DIR)

    # 2. Build 15 synthetic submissions
    # sub_01: 3-page PDF with headers/footers (source for exact copy + cross-modal MP4)
    sub01_p1 = (
        "Unstop Case Challenge 2026\n"
        "Executive Summary: Apex Consumer Appliances Vertical Integration Roadmap.\n"
        "Over the past four quarters, fragmented contract manufacturing eroded our gross margin on "
        "entry-level kitchen appliances by 290 basis points. Independent mold vendors charged steep "
        "rush surcharges whenever seasonal festival demand spiked above forecast.\n"
        "Page 1 of 3"
    )
    sub01_p2 = (
        "Unstop Case Challenge 2026\n"
        f"{CROSS_MODAL_PARAGRAPH}\n"
        "Page 2 of 3"
    )
    sub01_p3 = (
        "Unstop Case Challenge 2026\n"
        "Financial Impact: Capital expenditure of Rs 18 crore for in-house injection molding lines pays "
        "back within nineteen months through lower unit tooling costs and reduced warranty replacements.\n"
        "Page 3 of 3"
    )
    _write_pdf(SUBMISSIONS_DIR / "sub_01_vertical_orig.pdf", [sub01_p1, sub01_p2, sub01_p3])

    # sub_02: Exact copy of sub_01 in DOCX with Enter-spam anti-gaming test
    sub02_body = (
        "Over the past four quarters, fragmented contract manufacturing eroded our gross margin on "
        "entry-level kitchen appliances by 290 basis points. Independent mold vendors charged steep "
        "rush surcharges whenever seasonal festival demand spiked above forecast. "
        f"{CROSS_MODAL_PARAGRAPH} "
        "Capital expenditure of Rs 18 crore for in-house injection molding lines pays "
        "back within nineteen months through lower unit tooling costs and reduced warranty replacements."
    )
    # Deliberately fragment into short Enter-spammed paragraphs to prove anti-gaming chunking
    words = sub02_body.split()
    enter_spammed_paras = [" ".join(words[i : i + 6]) for i in range(0, len(words), 6)]
    _write_docx(SUBMISSIONS_DIR / "sub_02_vertical_copy.docx", enter_spammed_paras)

    # sub_03: Cross-modal MP4 reading sub_01 page 2 with spoken fillers (Scenario 1)
    logger.info("Synthesizing spoken MP4: sub_03_pitch_copied.mp4")
    _synthesize_spoken_mp4(SUBMISSIONS_DIR / "sub_03_pitch_copied.mp4", CROSS_MODAL_SPOKEN_SCRIPT)

    # sub_04 & sub_05: Paraphrased DOCX pair (Scenario 2)
    sub04_paras = [
        "Cold-Chain Spoilage Reduction Strategy for Perishable Dairy Distribution.",
        "Rural dairy cooperatives lose nearly eleven percent of chilled milk volume during summer transit "
        "because diesel refrigerated vans experience frequent compressor breakdowns on unpaved district roads.",
        "Installing phase-change thermal battery packs inside insulated collection crates maintains four-degree "
        "cooling for nine hours without requiring continuous engine power during roadside stops.",
        "Pairing these passive crates with IoT temperature tags enables automated rejection alerts before "
        "spoiled batches contaminate centralized pasteurization silos, lifting farmer payouts by eight percent.",
    ]
    sub05_paras = [
        "Minimizing Perishable Milk Wastage Across Rural Cold-Chain Logistics.",
        "Village milk unions forfeit close to eleven percent of cooled dairy stock throughout hot-weather "
        "hauling since diesel chiller trucks suffer repeated cooling-unit failures along rough rural highways.",
        "Equipping insulated transport containers with phase-change thermal gel batteries sustains four-degree "
        "refrigeration for nine straight hours even when truck engines are switched off at collection halts.",
        "Combining those passive cooling boxes with connected thermal sensors triggers instant spoilage warnings "
        "prior to tainted milk entering central processing tanks, boosting rural producer earnings by eight percent.",
    ]
    _write_docx(SUBMISSIONS_DIR / "sub_04_coldchain_a.docx", sub04_paras)
    _write_docx(SUBMISSIONS_DIR / "sub_05_coldchain_paraphrase.docx", sub05_paras)

    # sub_06: External plagiarism PDF copying from external_knowledge/ref_maritime_supply_chains.txt
    sub06_text = (
        "Maritime Import Risk Assessment for Industrial Machinery.\n"
        f"{EXTERNAL_ARTICLES['ref_maritime_supply_chains.txt']}\n"
        "Implementing these maritime hedging practices should reduce our quarterly customs penalty bill."
    )
    _write_pdf(SUBMISSIONS_DIR / "sub_06_external_wiki.pdf", [sub06_text])

    # sub_07: Clearly LLM-styled DOCX to test AI detection
    sub07_paras = [
        "Furthermore, adopting a holistic approach to enterprise transformation is paramount for sustaining "
        "competitive advantage across rapidly evolving regional markets.",
        "Moreover, it is crucial to note that seamless integration of predictive analytics plays a crucial role "
        "in optimizing resource allocation and strengthening operational resilience.",
        "Additionally, leadership teams must leverage synergies across cross-functional departments to establish "
        "a robust framework capable of navigating the complexities of modern supply networks.",
        "Consequently, implementing a comprehensive strategy grounded in data-driven governance underscores the "
        "multifaceted benefits of proactive risk mitigation and disciplined capital deployment.",
        "In conclusion, prioritizing scalable digital infrastructure will ultimately foster innovation and deliver "
        "enduring stakeholder value across the broader industrial landscape.",
    ]
    _write_docx(SUBMISSIONS_DIR / "sub_07_ai_generated.docx", sub07_paras)

    # sub_08 .. sub_12: 5 honest submissions sharing the EXACT 150-word prompt + unique answers (Scenario 4)
    honest_answers = [
        (
            "sub_08_honest_prompt_1.pdf",
            "pdf",
            "Our field survey across 22 Rajasthan weaving clusters showed power cuts cost looms Rs 9,500 weekly. "
            "We didn't pitch giant utility panels; instead, we leased 15 kW shared rooftop batteries to co-ops. "
            "Why? Because weavers can't afford upfront capex, and daytime sun matches peak loom hours! "
            "With UPI auto-debit at Rs 180 per shift, payback hits 26 months and loom uptime jumps from 61% to 94%.",
        ),
        (
            "sub_09_honest_prompt_2.pdf",
            "pdf",
            "Small kirana owners in Nagpur rarely default out of bad intent -- they miss payments when wholesale "
            "distributors demand cash right before festival week. Our working-capital line settles directly with "
            "FMCG wholesalers in 14-day micro-tranches instead of handing out lump-sum cash loans. "
            "We're testing WhatsApp voice reminders in Marathi, which already pushed on-time repayment to 98.1%!",
        ),
        (
            "sub_10_honest_prompt_3.pdf",
            "pdf",
            "Tomato farmers in Kolar lose 28% of harvest value simply because mandi auctions peak at 5 AM while "
            "grading trucks arrive late. By placing mobile optical-sorting tables right at village collection sheds, "
            "grade-A crates go straight to Bengaluru supermarkets while bruised fruit routes to puree processors. "
            "That simple split adds Rs 6 per kilo to farmgate income without building expensive cold warehouses.",
        ),
        (
            "sub_11_honest_prompt_4.docx",
            "docx",
            "Last-mile e-commerce riders in Jaipur hate waiting 4 hours at slow wall chargers, so they skip evening "
            "surge shifts. Switching our 300-scooter fleet to swappable LFP battery kiosks inside fuel stations cuts "
            "refuel downtime to 90 seconds! Yes, battery leasing costs 12% more per kilometer than home charging, "
            "but each rider completes 11 extra parcels daily, lifting net hub contribution margin by Rs 4.2 lakh/month.",
        ),
        (
            "sub_12_honest_prompt_5.docx",
            "docx",
            "Most vocational coding bootcamps fail in Tier-3 colleges because students watch 2-hour English lectures "
            "on patchy 4G and quit by week three. Our Hindi-English blended cohort uses 6-minute debugging drills "
            "running offline on cheap Android tablets, paired with weekend peer code reviews at local libraries. "
            "Completion rates climbed from 19% to 71%, and employer placement fees cover 85% of tutor salaries.",
        ),
    ]
    for fname, ftype, answer in honest_answers:
        if ftype == "pdf":
            _write_pdf(SUBMISSIONS_DIR / fname, [f"{SHARED_CASE_PROMPT}\n\n{answer}"])
        else:
            _write_docx(SUBMISSIONS_DIR / fname, [SHARED_CASE_PROMPT, answer])

    # sub_13: Original clean spoken MP4 (no false positive)
    logger.info("Synthesizing spoken MP4: sub_13_pitch_original.mp4")
    _synthesize_spoken_mp4(SUBMISSIONS_DIR / "sub_13_pitch_original.mp4", ORIGINAL_PITCH_SPOKEN_SCRIPT)

    # sub_14: 0-byte empty PDF (Scenario 3)
    (SUBMISSIONS_DIR / "sub_14_corrupt_empty.pdf").write_bytes(b"")

    # sub_15: Valid MP4 with silent audio track (Scenario 3)
    logger.info("Generating silent MP4: sub_15_silent_video.mp4")
    _create_silent_mp4(SUBMISSIONS_DIR / "sub_15_silent_video.mp4", duration_sec=5)

    # 3. Ground-Truth Manifest
    manifest: Dict[str, Any] = {
        "dataset_name": "Unstop Multimodal Integrity Benchmark v1",
        "total_files": 15,
        "format_counts": {"pdf": 6, "docx": 6, "mp4": 3},
        "data_compliance": "100% synthetic original business case studies and open-style reference summaries.",
        "files": [
            {
                "document_id": "sub_01_vertical_orig.pdf",
                "modality": "pdf",
                "expected_flagged": True,
                "expected_category": "peer_and_cross_modal",
                "related_files": ["sub_02_vertical_copy.docx", "sub_03_pitch_copied.mp4"],
                "notes": "3-page PDF with headers/footers; copied by sub_02 (exact) and sub_03 (MP4 speech on p.2).",
            },
            {
                "document_id": "sub_02_vertical_copy.docx",
                "modality": "docx",
                "expected_flagged": True,
                "expected_category": "peer_exact_copy",
                "related_files": ["sub_01_vertical_orig.pdf", "sub_03_pitch_copied.mp4"],
                "notes": "Exact copy of sub_01 fragmented with Enter-spam every 6 words to test anti-gaming chunking.",
            },
            {
                "document_id": "sub_03_pitch_copied.mp4",
                "modality": "mp4",
                "expected_flagged": True,
                "expected_category": "cross_modal_copy",
                "related_files": ["sub_01_vertical_orig.pdf", "sub_02_vertical_copy.docx"],
                "notes": "Scenario 1: Spoken English MP4 reading sub_01 p.2 with 'um/uh/ah' fillers.",
            },
            {
                "document_id": "sub_04_coldchain_a.docx",
                "modality": "docx",
                "expected_flagged": True,
                "expected_category": "peer_paraphrase",
                "related_files": ["sub_05_coldchain_paraphrase.docx"],
                "notes": "Scenario 2: Original cold-chain text heavily paraphrased by sub_05.",
            },
            {
                "document_id": "sub_05_coldchain_paraphrase.docx",
                "modality": "docx",
                "expected_flagged": True,
                "expected_category": "peer_paraphrase",
                "related_files": ["sub_04_coldchain_a.docx"],
                "notes": "Scenario 2: Synonym-swapped paraphrase of sub_04.",
            },
            {
                "document_id": "sub_06_external_wiki.pdf",
                "modality": "pdf",
                "expected_flagged": True,
                "expected_category": "external_plagiarism",
                "related_files": ["ref_maritime_supply_chains.txt"],
                "notes": "Copies external reference article from external_knowledge/.",
            },
            {
                "document_id": "sub_07_ai_generated.docx",
                "modality": "docx",
                "expected_flagged": True,
                "expected_category": "ai_generated",
                "related_files": [],
                "notes": "Uniform, formal LLM-styled prose to test AI-generation signal.",
            },
            {
                "document_id": "sub_08_honest_prompt_1.pdf",
                "modality": "pdf",
                "expected_flagged": False,
                "expected_category": "honest_shared_prompt",
                "related_files": [],
                "notes": "Scenario 4: Shares 150-word exam prompt + original solar microgrid answer.",
            },
            {
                "document_id": "sub_09_honest_prompt_2.pdf",
                "modality": "pdf",
                "expected_flagged": False,
                "expected_category": "honest_shared_prompt",
                "related_files": [],
                "notes": "Scenario 4: Shares 150-word exam prompt + original kirana fintech answer.",
            },
            {
                "document_id": "sub_10_honest_prompt_3.pdf",
                "modality": "pdf",
                "expected_flagged": False,
                "expected_category": "honest_shared_prompt",
                "related_files": [],
                "notes": "Scenario 4: Shares 150-word exam prompt + original agri-tech answer.",
            },
            {
                "document_id": "sub_11_honest_prompt_4.docx",
                "modality": "docx",
                "expected_flagged": False,
                "expected_category": "honest_shared_prompt",
                "related_files": [],
                "notes": "Scenario 4: Shares 150-word exam prompt + original EV fleet answer.",
            },
            {
                "document_id": "sub_12_honest_prompt_5.docx",
                "modality": "docx",
                "expected_flagged": False,
                "expected_category": "honest_shared_prompt",
                "related_files": [],
                "notes": "Scenario 4: Shares 150-word exam prompt + original ed-tech answer.",
            },
            {
                "document_id": "sub_13_pitch_original.mp4",
                "modality": "mp4",
                "expected_flagged": False,
                "expected_category": "honest_original_video",
                "related_files": [],
                "notes": "Original spoken MP4 pitch (rural healthcare ultrasound) to verify zero false positives.",
            },
            {
                "document_id": "sub_14_corrupt_empty.pdf",
                "modality": "pdf",
                "expected_flagged": False,
                "expected_category": "degraded_empty",
                "expected_status": "skipped_empty",
                "related_files": [],
                "notes": "Scenario 3: 0-byte empty PDF; must warn & skip without crashing.",
            },
            {
                "document_id": "sub_15_silent_video.mp4",
                "modality": "mp4",
                "expected_flagged": False,
                "expected_category": "degraded_silent_video",
                "expected_status": "skipped_no_audio",
                "related_files": [],
                "notes": "Scenario 3: Valid MP4 with silent audio track; must warn & skip without crashing.",
            },
        ],
    }

    manifest_path = SUBMISSIONS_DIR / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Generated 15 synthetic submissions + dataset_manifest.json in %s", SUBMISSIONS_DIR)


if __name__ == "__main__":
    generate_all()
