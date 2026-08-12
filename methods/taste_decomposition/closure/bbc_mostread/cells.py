#!/usr/bin/env python3
"""Minimal cell metadata for the bbc_mostread closure round scripts (audit.py
imports `cells as C` for CELL_META only). The full population loader lives in
round0_bbc.py / stage1_slice_bbc.py; nothing here loads data."""

CELL_META = {
    "bbc_mostread": dict(
        group_column="capture_day", item="news headline",
        corpus=("BBC News headlines: for each captured day, headlines that made the "
                "site's most-read module versus other same-day BBC headlines"),
        construct=("which headlines BBC readers actually click and read most"),
        text_trunc=300,
        layer1="bbc_mostread_layer1.json",
    ),
}
