# Deviations from the locked FM protocol (after the 13 November 2026 input lock)

Amendment A2 (`FM_AMENDMENTS.md` l.63) says that after the input lock the amendments never change
and later notes go here. This file is not hashed. Decisions and post hoc rules live in
`protocols.md` section 8; this file records only places where a locked text is wrong or cannot be
followed as written. No entry changes a locked number.

| # | Date | Locked text | What is wrong | How the paper handles it | Source |
|---|---|---|---|---|---|
| D1 | 20 Nov 2026 | A2 (`FM_AMENDMENTS.md` l.140) and A3's table (l.313) call the OCTID two-way test the test "where image size is constant" | OCTID images are not all one size: CSC 750–879 px wide, MH all 750 px; every OCTID image is larger than every benchmark CSC and MH development image, so the size-only reference sends all 204 queries to CSC and scores 0.50 | The paper says the size-only reference cannot separate the two classes in this test; it does not say image size is constant or that the test is free of shortcuts | Locked `reports/geometry/MIRAGE/geometry_summary.csv` and `external_two_way.csv` in `results-20261114T093033Z.zip`; found by the 19 Nov draft review (conflict C2) |
