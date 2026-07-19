from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def create_dummy_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    lines = [
        "SOP Pump P-101 - Safety Operating Procedure",
        "",
        "Purpose: This document defines safe start-up, inspection, and shutdown steps for pump P-101.",
        "1. Verify the suction and discharge valves are in the correct position.",
        "2. Apply Lock-Out Tag-Out before opening the coupling guard or touching the seal housing.",
        "3. Check for seal leakage, abnormal vibration, and bearing temperature above 85 C.",
        "4. If seal leakage is detected, stop the pump immediately and notify maintenance.",
        "5. Use PPE: gloves, goggles, face shield, and safety shoes.",
        "6. Confirm the motor current is within nameplate limits before returning to service.",
        "",
        "Hazards: rotating equipment, hot surfaces, pressure release, and entanglement.",
        "Equipment Tags: P-101, M-101, V-201, FT-104.",
    ]

    y = height - 72
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
        if y < 72:
            c.showPage()
            y = height - 72

    c.save()


def create_dummy_csv(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "Date": "2026-06-12",
                "Equipment_Tag": "P-101",
                "Issue": "Seal leakage detected during operation",
                "Inspection_Engineer": "A. Sharma",
                "Maintenance_Engineer": "K. Verma",
                "Inspection_Details": "Detected seal leakage during round inspection and isolated the pump.",
                "Maintenance_Details": "Replaced worn seal and completed alignment check before restart.",
                "Action_Taken": "Pump isolated, seal inspected, seal replaced, and restart checks completed",
            },
            {
                "Date": "2026-06-13",
                "Equipment_Tag": "P-101",
                "Issue": "Bearing temperature exceeded threshold",
                "Inspection_Engineer": "R. Kulkarni",
                "Maintenance_Engineer": "M. Iyer",
                "Inspection_Details": "Recorded bearing temperature at 92 C and flagged overheating.",
                "Maintenance_Details": "Performed lubrication, corrected misalignment, and verified normal temperature.",
                "Action_Taken": "Bearing lubrication and alignment completed",
            },
            {
                "Date": "2026-06-14",
                "Equipment_Tag": "V-303",
                "Issue": "Hydrostatic test overdue",
                "Inspection_Engineer": "S. Nair",
                "Maintenance_Engineer": "P. Rao",
                "Inspection_Details": "Audit found no recent hydrostatic test certificate for vessel V-303.",
                "Maintenance_Details": "Raised hydrostatic test work order and scheduled certified testing team.",
                "Action_Taken": "Testing request raised for compliance closure",
            },
        ]
    )
    df.to_csv(output_path, index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    create_dummy_pdf(DATA_DIR / "SOP_Pump_P-101.pdf")
    create_dummy_csv(DATA_DIR / "Maintenance_Log.csv")
    print(f"Dummy data created in: {DATA_DIR}")


if __name__ == "__main__":
    main()
